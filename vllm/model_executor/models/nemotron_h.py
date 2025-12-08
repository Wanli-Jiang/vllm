# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# Adapted from https://github.com/vllm-project/vllm/blob/94d8ec8d2bcb4ec55e33022b313c7e978edf05e1/vllm/model_executor/models/bamba.py
# Copyright 2024 HuggingFace Inc. team. All rights reserved.
# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Inference-only NemotronH model."""

import typing
from collections.abc import Callable, Iterable
from itertools import islice

import torch
from torch import nn

from vllm.attention.layer import Attention
from vllm.compilation.decorators import support_torch_compile
from vllm.config import CacheConfig, ModelConfig, VllmConfig
from vllm.config.parallel import ParallelConfig
from vllm.distributed import get_ep_group, get_tensor_model_parallel_world_size
from vllm.distributed.communication_op import tensor_model_parallel_all_gather
from vllm.distributed.parallel_state import get_pp_group
from vllm.model_executor.layers.activation import ReLUSquaredActivation
from vllm.model_executor.layers.fused_moe import FusedMoE, SharedFusedMoE
from vllm.model_executor.layers.fused_moe.utils import activation_without_mul
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import (
    ColumnParallelLinear,
    QKVParallelLinear,
    ReplicatedLinear,
    RowParallelLinear,
)
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.mamba.mamba_mixer2 import MambaMixer2
from vllm.model_executor.layers.mamba.mamba_utils import (
    MambaStateDtypeCalculator,
    MambaStateShapeCalculator,
)
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from vllm.model_executor.model_loader.weight_utils import (
    default_weight_loader,
    maybe_remap_kv_scale_name,
)
from vllm.model_executor.models.interfaces import (
    HasInnerState,
    IsHybrid,
    MixtureOfExperts,
    SupportsLoRA,
    SupportsMambaPrefixCaching,
    SupportsPP,
    SupportsQuant,
)
from vllm.model_executor.models.utils import (
    AutoWeightsLoader,
    WeightsMapper,
    is_pp_missing_parameter,
    make_empty_intermediate_tensors_factory,
    make_layers,
    maybe_prefix,
    sequence_parallel_chunk,
)
from vllm.sequence import IntermediateTensors
from vllm.transformers_utils.configs import NemotronHConfig


SHOW_FLAG = False
LAYER_IDX = 0

class NemotronHMLP(nn.Module):
    def __init__(
        self,
        config: NemotronHConfig,
        intermediate_size: int,
        quant_config: QuantizationConfig | None = None,
        bias: bool = False,
        reduce_results: bool = True,
        is_sequence_parallel: bool = False,
        prefix: str = "",
    ) -> None:
        super().__init__()

        self.up_proj = ColumnParallelLinear(
            input_size=config.hidden_size,
            output_size=intermediate_size,
            bias=bias,
            quant_config=quant_config,
            disable_tp=is_sequence_parallel,
            prefix=f"{prefix}.up_proj",
            slice_padding=False,
        )
        self.down_proj = RowParallelLinear(
            input_size=intermediate_size,
            output_size=config.hidden_size,
            bias=bias,
            quant_config=quant_config,
            reduce_results=reduce_results,
            disable_tp=is_sequence_parallel,
            prefix=f"{prefix}.down_proj",
            slice_padding=False,
        )
        self.act_fn = ReLUSquaredActivation()

    def forward(self, x: torch.Tensor):
        x, _ = self.up_proj(x)
        x = self.act_fn(x)
        x, _ = self.down_proj(x)
        return x


class NemotronHMoE(nn.Module):
    def __init__(
        self,
        config: NemotronHConfig,
        quant_config: QuantizationConfig | None = None,
        parallel_config: ParallelConfig | None = None,
        prefix: str = "",
    ):
        super().__init__()
        self.tp_size = get_tensor_model_parallel_world_size()
        self.routed_scaling_factor = config.routed_scaling_factor

        self.ep_group = get_ep_group().device_group
        self.ep_rank = self.ep_group.rank()
        self.ep_size = self.ep_group.size()
        self.n_routed_experts: int = config.n_routed_experts
        self.n_shared_experts: int = config.n_shared_experts

        self.is_sequence_parallel = parallel_config.use_sequence_parallel_moe

        self.gate = ReplicatedLinear(
            config.hidden_size,
            config.n_routed_experts,
            bias=False,
            params_dtype=torch.float32,
            quant_config=None,
            prefix=f"{prefix}.gate",
        )

        self.gate.e_score_correction_bias = nn.Parameter(
            torch.empty(config.n_routed_experts, dtype=torch.float32)
        )
        # Load balancing settings.
        self.enable_eplb = parallel_config.enable_eplb

        self.n_redundant_experts = parallel_config.eplb_config.num_redundant_experts  # noqa: E501
        self.n_logical_experts = self.n_routed_experts
        self.n_physical_experts = self.n_logical_experts + self.n_redundant_experts
        self.n_local_physical_experts = self.n_physical_experts // self.ep_size

        self.physical_expert_start = self.ep_rank * self.n_local_physical_experts
        self.physical_expert_end = (
            self.physical_expert_start + self.n_local_physical_experts
        )

        if config.n_shared_experts is None or config.n_shared_experts == 0:
            self.shared_experts = None
        else:
            intermediate_size = (
                config.moe_shared_expert_intermediate_size * config.n_shared_experts
            )

            self.shared_experts = NemotronHMLP(
                config=config,
                intermediate_size=intermediate_size,
                quant_config=quant_config,
                reduce_results=False,
                is_sequence_parallel=self.is_sequence_parallel,
                prefix=f"{prefix}.shared_experts",
            )

        self.experts = SharedFusedMoE(
            shared_experts=self.shared_experts,
            num_experts=config.n_routed_experts,
            top_k=config.num_experts_per_tok,
            hidden_size=config.hidden_size,
            intermediate_size=config.moe_intermediate_size,
            reduce_results=False,
            renormalize=config.norm_topk_prob,
            quant_config=quant_config,
            use_grouped_topk=True,
            num_expert_group=config.n_group,
            topk_group=config.topk_group,
            prefix=f"{prefix}.experts",
            scoring_func="sigmoid",
            e_score_correction_bias=self.gate.e_score_correction_bias,
            activation=activation_without_mul(config.mlp_hidden_act),
            is_act_and_mul=False,  # non-gated MoE
            enable_eplb=self.enable_eplb,
            num_redundant_experts=self.n_redundant_experts,
            is_sequence_parallel=self.is_sequence_parallel,
        )

    def forward(self, hidden_states: torch.Tensor, show: bool = False) -> torch.Tensor:
        num_tokens, hidden_dim = hidden_states.shape
        hidden_states = hidden_states.view(-1, hidden_dim)

        if self.is_sequence_parallel:
            hidden_states = sequence_parallel_chunk(hidden_states)

        # router_logits: (num_tokens, n_experts)

        if show:
            print("="*100)
            print(f"Layer {LAYER_IDX} gate input: {hidden_states.shape=!r} {hidden_states.dtype=!r} {hidden_states.device=!r} {hidden_states=!r}")
            print("="*100)

        router_logits, _ = self.gate(hidden_states.to(dtype=torch.float32))

        if show:
            print("="*100)
            print(f"Layer {LAYER_IDX} gate output: {router_logits.shape=!r} {router_logits.dtype=!r} {router_logits.device=!r} {router_logits=!r}")
            print("="*100)

        fused_moe_out = self.experts(
            hidden_states=hidden_states, router_logits=router_logits, show=show,
        )

        shared_output, final_hidden_states = fused_moe_out

        if show:
            print("="*100)
            print(f"Layer {LAYER_IDX} shared_output output: {shared_output.shape=!r} {shared_output.dtype=!r} {shared_output.device=!r} {shared_output=!r}")
            print(f"Layer {LAYER_IDX} final_hidden_states output: {final_hidden_states.shape=!r} {final_hidden_states.dtype=!r} {final_hidden_states.device=!r} {final_hidden_states=!r}")
            print("="*100)

        # Fix FP16 overflow
        # See DeepseekV2DecoderLayer for more details.
        if hidden_states.dtype != torch.float16:
            final_hidden_states *= self.routed_scaling_factor
        elif self.shared_experts is not None:
            assert shared_output is not None
            shared_output *= 1.0 / self.routed_scaling_factor

        if self.shared_experts is not None:
            assert shared_output is not None
            final_hidden_states += shared_output

        if self.is_sequence_parallel:
            final_hidden_states = tensor_model_parallel_all_gather(
                final_hidden_states, 0
            )
            final_hidden_states = final_hidden_states[:num_tokens]
        elif self.tp_size > 1:
            final_hidden_states = self.experts.maybe_all_reduce_tensor_model_parallel(
                final_hidden_states
            )

        return final_hidden_states.view(num_tokens, hidden_dim)


class NemotronHMLPDecoderLayer(nn.Module):
    def __init__(
        self,
        config: NemotronHConfig,
        layer_idx: int,
        model_config: ModelConfig | None = None,
        cache_config: CacheConfig | None = None,
        quant_config: QuantizationConfig | None = None,
        parallel_config: ParallelConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.config = config

        hybrid_override_pattern = config.hybrid_override_pattern
        mlp_index = hybrid_override_pattern[: layer_idx + 1].count("-") - 1
        if isinstance(config.intermediate_size, list):
            if len(config.intermediate_size) == 1:
                intermediate_size = config.intermediate_size[0]
            else:
                intermediate_size = config.intermediate_size[mlp_index]
        else:
            intermediate_size = config.intermediate_size

        self.mixer = NemotronHMLP(
            config,
            intermediate_size=intermediate_size,
            quant_config=quant_config,
            bias=config.mlp_bias,
            prefix=f"{prefix}.mixer",
        )

        self.norm = RMSNorm(config.hidden_size, eps=config.layer_norm_epsilon)

    def forward(
        self,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
        **kwargs,
    ):
        if residual is None:
            residual = hidden_states
            hidden_states = self.norm(hidden_states)
        else:
            hidden_states, residual = self.norm(hidden_states, residual)

        hidden_states = self.mixer(hidden_states)
        return hidden_states, residual


class NemotronHMoEDecoderLayer(nn.Module):
    def __init__(
        self,
        config: NemotronHConfig,
        layer_idx: int,
        model_config: ModelConfig | None = None,
        cache_config: CacheConfig | None = None,
        quant_config: QuantizationConfig | None = None,
        parallel_config: ParallelConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.config = config

        self.mixer = NemotronHMoE(
            config,
            quant_config=quant_config,
            parallel_config=parallel_config,
            prefix=f"{prefix}.mixer",
        )

        self.norm = RMSNorm(config.hidden_size, eps=config.layer_norm_epsilon)

    def forward(
        self,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
        **kwargs,
    ):
        if residual is None:
            residual = hidden_states
            hidden_states = self.norm(hidden_states)
        else:
            hidden_states, residual = self.norm(hidden_states, residual)

        show = kwargs["show"]

        if show:
            print("="*100)
            print(f"Layer {LAYER_IDX} input: {hidden_states.shape=!r} {hidden_states.dtype=!r} {hidden_states.device=!r} {hidden_states=!r}")
            print("="*100)

        hidden_states = self.mixer(hidden_states, show=show)

        if show:
            print("="*100)
            print(f"Layer {LAYER_IDX} output: {hidden_states.shape=!r} {hidden_states.dtype=!r} {hidden_states.device=!r} {hidden_states=!r}")
            print("="*100)

        return hidden_states, residual


def dequantize_fp8_pb_wo_to_bf16(weight: torch.Tensor,
                                 weight_scale: torch.Tensor,
                                 block_m: int = 128,
                                 block_n: int = 128) -> torch.Tensor:
    """
    Dequantize ModelOpt fp8_pb_wo weights to bfloat16.
    weight:       [M, N] FP8 tensor
    weight_scale: [M/block_m, N/block_n] scale tensor
    Returns:      [M, N] BF16 tensor
    """
    M, N = weight.shape

    print(f"dequantize_fp8_pb_wo_to_bf16: {weight.shape=!r} {weight_scale.shape=!r}")

    assert M % block_m == 0 and N % block_n == 0, "Incompatible block size"
    num_blocks_m = M // block_m
    num_blocks_n = N // block_n
    assert weight_scale.shape == (num_blocks_m, num_blocks_n)

    # Reshape weight into blocks: [num_blocks_m, block_m, num_blocks_n, block_n]
    w_blocks = weight.view(num_blocks_m, block_m, num_blocks_n, block_n)

    # Broadcast scales to block shape: [num_blocks_m, 1, num_blocks_n, 1]
    scales = weight_scale.to(torch.bfloat16).view(num_blocks_m, 1, num_blocks_n, 1)

    # Dequantize: fp8 -> bf16, then multiply by scale
    w_dequant = w_blocks.to(torch.bfloat16) * scales

    # Reshape back to [M, N]
    w_dequant = w_dequant.view(M, N)

    return w_dequant


class NemotronHMambaDecoderLayer(nn.Module):
    def __init__(
        self,
        config: NemotronHConfig,
        layer_idx: int,
        model_config: ModelConfig | None = None,
        cache_config: CacheConfig | None = None,
        quant_config: QuantizationConfig | None = None,
        parallel_config: ParallelConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.config = config
        self.mixer = MambaMixer2(
            hidden_size=config.hidden_size,
            ssm_state_size=config.ssm_state_size,
            conv_kernel_size=config.conv_kernel,
            intermediate_size=config.mamba_num_heads * config.mamba_head_dim,
            use_conv_bias=config.use_conv_bias,
            use_bias=config.use_bias,
            n_groups=config.n_groups,
            num_heads=config.mamba_num_heads,
            head_dim=config.mamba_head_dim,
            rms_norm_eps=config.layer_norm_epsilon,
            activation=config.mamba_hidden_act,
            model_config=model_config,
            cache_config=cache_config,
            quant_config=quant_config,
            prefix=f"{prefix}.mixer",
        )

        self.norm = RMSNorm(config.hidden_size, eps=config.layer_norm_epsilon)

    def forward(
        self,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
        **kwargs,
    ):
        if residual is None:
            residual = hidden_states
            hidden_states = self.norm(hidden_states)
        else:
            hidden_states, residual = self.norm(hidden_states, residual)

        show = kwargs["show"]

        if show:
            print("="*100)
            print(f"Layer {LAYER_IDX} input: {hidden_states.shape=!r} {hidden_states.dtype=!r} {hidden_states.device=!r} {hidden_states=!r}")
            print("="*100)

        # if show:
        #     print("="*100)
        #     # Print all parameter weights for self.mixer
        #     print(f"Layer {LAYER_IDX} mixer parameter weights:")
        #     in_proj_weight, in_proj_weight_scale = None, None
        #     for name, param in self.mixer.named_parameters():
        #         print(f"  {name}: {param.shape} dtype={param.dtype} device={param.device} values: {param.data}")

        #         if name == "in_proj.weight":
        #             in_proj_weight = param
        #         elif name == "in_proj.weight_scale":
        #             in_proj_weight_scale = param

        #     if in_proj_weight is not None and in_proj_weight_scale is not None:
        #         in_proj_weight_dq = dequantize_fp8_pb_wo_to_bf16(in_proj_weight, in_proj_weight_scale)
        #         print(f"Layer {LAYER_IDX} in_proj_dequantized: {in_proj_weight_dq.shape=!r} {in_proj_weight_dq.dtype=!r} {in_proj_weight_dq.device=!r} {in_proj_weight_dq=!r}")
        #     print("="*100)

        output = self.mixer(hidden_states, show=show)

        # raise InterruptedError(f"Stop here {LAYER_IDX=!r} {show=!r}")

        if show:
            print("="*100)
            print(f"Layer {LAYER_IDX} output: {output.shape=!r} {output.dtype=!r} {output.device=!r} {output=!r}")
            print("="*100)

        return output, residual


class NemotronHAttention(nn.Module):
    def __init__(
        self,
        config: NemotronHConfig,
        layer_idx: int,
        model_config: ModelConfig | None = None,
        cache_config: CacheConfig | None = None,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size
        tp_size = get_tensor_model_parallel_world_size()
        self.total_num_heads = config.num_attention_heads
        assert self.total_num_heads % tp_size == 0
        self.num_heads = self.total_num_heads // tp_size
        self.total_num_kv_heads = config.num_key_value_heads
        if self.total_num_kv_heads >= tp_size:
            # Number of KV heads is greater than TP size, so we partition
            # the KV heads across multiple tensor parallel GPUs.
            assert self.total_num_kv_heads % tp_size == 0
        else:
            # Number of KV heads is less than TP size, so we replicate
            # the KV heads across multiple tensor parallel GPUs.
            assert tp_size % self.total_num_kv_heads == 0
        self.num_kv_heads = max(1, self.total_num_kv_heads // tp_size)
        if hasattr(config, "head_dim") and config.head_dim is not None:
            self.head_dim = config.head_dim
        else:
            self.head_dim = config.hidden_size // self.total_num_heads
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim**-0.5

        self.qkv_proj = QKVParallelLinear(
            config.hidden_size,
            self.head_dim,
            self.total_num_heads,
            self.total_num_kv_heads,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.qkv_proj",
        )
        self.o_proj = RowParallelLinear(
            self.total_num_heads * self.head_dim,
            config.hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.o_proj",
        )

        self.attn = Attention(
            self.num_heads,
            self.head_dim,
            self.scaling,
            num_kv_heads=self.num_kv_heads,
            cache_config=cache_config,
            prefix=f"{prefix}.attn",
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        attn_output = self.attn(q, k, v)
        output, _ = self.o_proj(attn_output)
        return output


class NemotronHAttentionDecoderLayer(nn.Module):
    def __init__(
        self,
        config: NemotronHConfig,
        layer_idx: int,
        model_config: ModelConfig | None = None,
        cache_config: CacheConfig | None = None,
        quant_config: QuantizationConfig | None = None,
        parallel_config: ParallelConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()

        self.mixer = NemotronHAttention(
            config,
            layer_idx,
            model_config,
            cache_config,
            quant_config,
            prefix=f"{prefix}.mixer",
        )

        self.norm = RMSNorm(config.hidden_size, eps=config.layer_norm_epsilon)

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
        **kwargs,
    ):
        if residual is None:
            residual = hidden_states
            hidden_states = self.norm(hidden_states)
        else:
            hidden_states, residual = self.norm(hidden_states, residual)

        hidden_states = self.mixer(hidden_states=hidden_states)
        return hidden_states, residual


ALL_DECODER_LAYER_TYPES = {
    "M": NemotronHMambaDecoderLayer,
    "-": NemotronHMLPDecoderLayer,
    "*": NemotronHAttentionDecoderLayer,
    "E": NemotronHMoEDecoderLayer,
}


@support_torch_compile
class NemotronHModel(nn.Module):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()

        config: NemotronHConfig = vllm_config.model_config.hf_config
        model_config = vllm_config.model_config
        cache_config = vllm_config.cache_config
        quant_config = vllm_config.quant_config
        parallel_config = vllm_config.parallel_config

        self.config = config

        self.vocab_size = config.vocab_size

        self.embed_tokens = VocabParallelEmbedding(
            self.vocab_size,
            config.hidden_size,
        )

        self.has_moe = "E" in config.hybrid_override_pattern

        def get_layer(prefix: str):
            layer_idx = int(prefix.rsplit(".", 1)[1])
            layer_class = ALL_DECODER_LAYER_TYPES[
                config.hybrid_override_pattern[layer_idx]
            ]
            return layer_class(
                config=config,
                layer_idx=layer_idx,
                model_config=model_config,
                cache_config=cache_config,
                quant_config=quant_config,
                parallel_config=parallel_config,
                prefix=prefix,
            )

        self.start_layer, self.end_layer, self.layers = make_layers(
            len(config.hybrid_override_pattern), get_layer, prefix=f"{prefix}.layers"
        )
        self.make_empty_intermediate_tensors = make_empty_intermediate_tensors_factory(
            ["hidden_states", "residual"], config.hidden_size
        )

        self.norm_f = RMSNorm(config.hidden_size, eps=config.layer_norm_epsilon)

        if SHOW_FLAG:
            print("="*100)
            print(f"{self.start_layer=!r}")
            print(f"{self.end_layer=!r}")
            print(f"{self.layers=!r}")
            print("="*100)

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_tokens(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor | IntermediateTensors:
        if get_pp_group().is_first_rank:
            if inputs_embeds is not None:
                hidden_states = inputs_embeds
            else:
                hidden_states = self.embed_input_ids(input_ids)
            residual = None
        else:
            assert intermediate_tensors is not None
            hidden_states = intermediate_tensors["hidden_states"]
            residual = intermediate_tensors["residual"]

        for layer_idx, layer in enumerate(islice(self.layers, self.start_layer, self.end_layer)):


            # if layer_idx == LAYER_IDX:
            #     print("="*100)
            #     print(f"layer_idx: {layer_idx=!r} {LAYER_IDX=!r} {SHOW_FLAG=!r} {hidden_states.shape=!r}")
            #     print("="*100)
            show = layer_idx == LAYER_IDX and SHOW_FLAG and hidden_states.shape[0] == 5

            # print("="*100)
            # print(f"Layer {layer_idx} input: {hidden_states.shape=!r}")
            # print("="*100)
            hidden_states, residual = layer(
                positions=positions,
                hidden_states=hidden_states,
                residual=residual,
                show=show,
            )

        if not get_pp_group().is_last_rank:
            return IntermediateTensors(
                {"hidden_states": hidden_states, "residual": residual}
            )
        hidden_states, _ = self.norm_f(hidden_states, residual)
        return hidden_states

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            ("qkv_proj", "q_proj", "q"),
            ("qkv_proj", "k_proj", "k"),
            ("qkv_proj", "v_proj", "v"),
        ]

        if self.has_moe:
            # (param_name, weight_name, expert_id, shard_id)
            expert_params_mapping = FusedMoE.make_expert_params_mapping(
                # - FusedMoe.w1 (aka gate_proj) should be up_proj since that's
                #   what the activation is applied to
                # - FusedMoe.w3 (aka up_proj) should be ignored since we're
                #   using non-gated MoE
                ckpt_gate_proj_name="up_proj",
                ckpt_down_proj_name="down_proj",
                ckpt_up_proj_name="",
                num_experts=self.config.n_routed_experts,
                num_redundant_experts=getattr(self, "num_redundant_experts", 0),
            )
        else:
            expert_params_mapping = []

        params_dict = dict(self.named_parameters())


        # print("="*100)
        # print(f" expert_params_mapping: {expert_params_mapping=!r}")
        # for k, v in params_dict.items():
        #     if "layers.20." in k:
        #         print(f" {k=!r} {v.shape=!r} {v.dtype=!r} {v.device=!r}")
        # print("="*100)

        loaded_params: set[str] = set()
        for name, loaded_weight in weights:

            # show = "layers.20." in name
            # if show:
            #     print("="*100)
            #     print(f" 1: {name=!r} {loaded_weight.shape=!r} {loaded_weight.dtype=!r} {loaded_weight.device=!r}")


            if "scale" in name:
                # Remapping the name of FP8 kv-scale.
                name = maybe_remap_kv_scale_name(name, params_dict)
                if name is None:
                    continue

            # if show:
            #     print(f" after maybe_remap_kv_scale_name: {name=!r} {loaded_weight.shape=!r} {loaded_weight.dtype=!r} {loaded_weight.device=!r}")

            # load stacked params
            for param_name, weight_name, shard_id in stacked_params_mapping:
                if weight_name not in name:
                    continue
                name = name.replace(weight_name, param_name)
                # Skip loading extra bias for GPTQ models.
                if name.endswith(".bias") and name not in params_dict:
                    continue

                if is_pp_missing_parameter(name, self):
                    continue

                param = params_dict[name]
                weight_loader = param.weight_loader
                weight_loader(param, loaded_weight, shard_id)

                # print("="*100)
                # print(f" PASS loaded weight: {name=!r} {loaded_weight.shape=!r} {param.shape=!r} {weight_loader=!r}")
                # print("="*100)

                break

            # load other params
            else:
                is_expert_weight = False
                for mapping in expert_params_mapping:
                    param_name, weight_name, expert_id, shard_id = mapping
                    if weight_name not in name:
                        continue

                    # if show:
                    #     print(f" 2: {name=!r} {weight_name=!r} {param_name=!r} {expert_id=!r} {shard_id=!r}")

                    # Anyway, this is an expert weight and should not be
                    # attempted to load as other weights later
                    is_expert_weight = True

                    # Do not modify `name` since the loop may continue here
                    # Instead, create a new variable
                    name_mapped = name.replace(weight_name, param_name)

                    # if show:
                    #     print(f" 3: {name_mapped=!r}")

                    if is_pp_missing_parameter(name_mapped, self):
                        continue
                    param = params_dict[name_mapped]
                    # We should ask the weight loader to return success or not
                    # here since otherwise we may skip experts with other
                    # available replicas.
                    weight_loader = typing.cast(
                        Callable[..., bool], param.weight_loader
                    )
                    success = weight_loader(
                        param,
                        loaded_weight,
                        name_mapped,
                        shard_id=shard_id,
                        expert_id=expert_id,
                        return_success=True,
                    )
                    # print("="*100)
                    # print(f" PASS loaded weight: {name=!r} {loaded_weight.shape=!r} {param.shape=!r} {weight_loader=!r}")
                    # print("="*100)
                    if success:
                        name = name_mapped
                        break
                else:
                    if is_expert_weight:
                        continue

                    if is_pp_missing_parameter(name, self):
                        continue

                    param = params_dict[name]
                    weight_loader = getattr(
                        param, "weight_loader", default_weight_loader
                    )
                    if "in_proj.weight_scale" in name and len(loaded_weight.shape) > 1:
                        # mamba in_proj is kept in a single weight while it should included multiple output sizes.
                        # pass None and it will fetch weights by param output_sizes.
                        weight_loader(param, loaded_weight, None)

                        # print("="*100)
                        # print(f" PASS loaded weight: {name=!r} {loaded_weight.shape=!r} {param.shape=!r} {weight_loader=!r}")
                        # print("="*100)
                    else:
                        weight_loader(param, loaded_weight)
                        # print("="*100)
                        # print(f" PASS loaded weight: {name=!r} {loaded_weight.shape=!r} {param.shape=!r} {weight_loader=!r}")
                        # print("="*100)

            loaded_params.add(name)
        return loaded_params


class NemotronHForCausalLM(
    nn.Module,
    HasInnerState,
    SupportsLoRA,
    SupportsPP,
    IsHybrid,
    SupportsQuant,
    MixtureOfExperts,
    SupportsMambaPrefixCaching,
):
    hf_to_vllm_mapper = WeightsMapper(
        orig_to_new_prefix={"backbone": "model"},
        orig_to_new_substr={"A_log": "A", "embeddings": "embed_tokens"},
    )

    packed_modules_mapping = {
        "qkv_proj": [
            "q_proj",
            "k_proj",
            "v_proj",
        ],
    }

    # LoRA specific attributes
    embedding_modules = {
        "embed_tokens": "input_embeddings",
        "lm_head": "output_embeddings",
    }
    embedding_padding_modules = ["lm_head"]

    @classmethod
    def get_mamba_state_dtype_from_config(
        cls,
        vllm_config: "VllmConfig",
    ) -> tuple[torch.dtype, torch.dtype]:
        return MambaStateDtypeCalculator.mamba2_state_dtype(
            vllm_config.model_config.dtype,
            vllm_config.cache_config.mamba_cache_dtype,
            vllm_config.cache_config.mamba_ssm_cache_dtype,
        )

    @classmethod
    def get_mamba_state_shape_from_config(
        cls,
        vllm_config: "VllmConfig",
    ) -> tuple[tuple[int, int], tuple[int, int, int]]:
        """Calculate shapes for Mamba's convolutional and state caches.

        Args:
            vllm_config: vLLM config

        Returns:
            Tuple containing:
            - conv_state_shape: Shape for convolutional state cache
            - temporal_state_shape: Shape for state space model cache
        """
        parallel_config = vllm_config.parallel_config
        hf_config = vllm_config.model_config.hf_config
        intermediate_size = hf_config.mamba_num_heads * hf_config.mamba_head_dim

        return MambaStateShapeCalculator.mamba2_state_shape(
            intermediate_size=intermediate_size,
            tp_world_size=parallel_config.tensor_parallel_size,
            n_groups=hf_config.n_groups,
            num_heads=hf_config.mamba_num_heads,
            head_dim=hf_config.mamba_head_dim,
            state_size=hf_config.ssm_state_size,
            conv_kernel=hf_config.conv_kernel,
        )

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        config = vllm_config.model_config.hf_config
        self.vllm_config = vllm_config
        self.model_config = vllm_config.model_config

        scheduler_config = vllm_config.scheduler_config

        self.quant_config = vllm_config.quant_config

        super().__init__()
        self.config = config
        self.scheduler_config = scheduler_config
        self.model = NemotronHModel(
            vllm_config=vllm_config, prefix=maybe_prefix(prefix, "model")
        )

        self.lm_head = ParallelLMHead(
            config.vocab_size,
            config.hidden_size,
            prefix=maybe_prefix(prefix, "lm_head"),
        )

        self.logits_processor = LogitsProcessor(config.vocab_size)

        self.make_empty_intermediate_tensors = (
            self.model.make_empty_intermediate_tensors
        )

        # Set MoE hyperparameters
        if self.model.has_moe:
            self.expert_weights = []
            self.num_expert_groups = config.n_group

            self.moe_layers = []
            example_moe = None
            for layer in self.model.layers:
                if isinstance(layer, NemotronHMoEDecoderLayer):
                    # Pick last one layer since the first ones
                    # may be dense layers.
                    example_moe = layer.mixer
                    self.moe_layers.append(layer.mixer.experts)

            self.num_moe_layers = len(self.moe_layers)
            self.num_logical_experts = example_moe.n_logical_experts
            self.num_physical_experts = example_moe.n_physical_experts
            self.num_local_physical_experts = example_moe.n_local_physical_experts  # noqa: E501
            self.num_routed_experts = example_moe.n_routed_experts
            self.num_shared_experts = example_moe.n_shared_experts
            self.num_redundant_experts = example_moe.n_redundant_experts

    def update_physical_experts_metadata(
        self,
        num_physical_experts: int,
        num_local_physical_experts: int,
    ) -> None:
        assert self.num_local_physical_experts == num_local_physical_experts
        self.num_physical_experts = num_physical_experts
        self.num_local_physical_experts = num_local_physical_experts
        self.num_redundant_experts = num_physical_experts - self.num_logical_experts
        for layer in self.model.layers:
            if isinstance(layer, NemotronHMoEDecoderLayer):
                moe = layer.mixer
                moe.n_local_physical_experts = num_local_physical_experts
                moe.n_physical_experts = num_physical_experts
                moe.n_redundant_experts = self.num_redundant_experts
                moe.experts.update_expert_map()

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.embed_input_ids(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        **kwargs,
    ):
        hidden_states = self.model(
            input_ids, positions, intermediate_tensors, inputs_embeds
        )

        return hidden_states

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor | None:
        logits = self.logits_processor(self.lm_head, hidden_states)
        return logits

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        loader = AutoWeightsLoader(self)
        return loader.load_weights(weights, mapper=self.hf_to_vllm_mapper)
