# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch

from vllm.distributed import (
    get_tensor_model_parallel_world_size,
    tensor_model_parallel_all_reduce,
)
from vllm.model_executor.layers.fused_moe.layer import FusedMoE


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



# TODO(bnell): Add shared + fused combo function? e.g. +
class SharedFusedMoE(FusedMoE):
    """
    A FusedMoE operation that also computes the results of shared experts.
    If an all2all communicator is being used the shared expert computation
    can be interleaved with the fused all2all dispatch communication step.
    """

    def __init__(
        self,
        shared_experts: torch.nn.Module | None,
        gate: torch.nn.Module | None = None,
        use_overlapped: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._shared_experts = shared_experts

        # Disable shared expert overlap if:
        #   - we are using eplb, because of correctness issues
        #   - we are using flashinfer with DP, since there nothint to gain
        #   - we are using marlin kjernels
        self.use_overlapped = (
            use_overlapped
            and not (
                # TODO(wentao): find the root cause and remove this condition
                self.enable_eplb
                or (self.moe_config.use_flashinfer_cutlass_kernels and self.dp_size > 1)
                or self.use_marlin_kernels
            )
            and self._shared_experts is not None
        )

        self._gate = gate

    @property
    def shared_experts(self) -> torch.nn.Module | None:
        return self._shared_experts if self.use_overlapped else None

    @property
    def gate(self) -> torch.nn.Module | None:
        return self._gate if self.use_overlapped else None

    @property
    def is_internal_router(self) -> bool:
        return self.gate is not None

    def forward(
        self,
        hidden_states: torch.Tensor,
        router_logits: torch.Tensor,
        show: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:

        # print("="*100)
        # print(f"show: {self.use_overlapped=!r}")
        # print(f"shared_experts: {self._shared_experts=!r}")
        # print(f"reduce_results: {self.reduce_results=!r}")
        # print(f"must_reduce_shared_expert_outputs: {self.must_reduce_shared_expert_outputs()=!r}")
        # print(f"get_tensor_model_parallel_world_size(): {get_tensor_model_parallel_world_size()=!r}")
        # print(f"tensor_model_parallel_all_reduce: {tensor_model_parallel_all_reduce=!r}")
        # print("="*100)
        # raise IOError("stop here")

        if not self.use_overlapped:
            if self._shared_experts is not None:
                shared_out = self._shared_experts(hidden_states)

                # Reduce shared expert outputs if necessary, since the MLP
                # should have been created with reduce_results=False.
                if (
                    self.reduce_results
                    and get_tensor_model_parallel_world_size() > 1
                    and self.must_reduce_shared_expert_outputs()
                ):
                    shared_out = tensor_model_parallel_all_reduce(shared_out)
            else:
                shared_out = None

            if show:
                print("="*100)
                up_proj_weight, up_proj_weight_scale = None, None
                down_proj_weight, down_proj_weight_scale = None, None
                for k, v in self._shared_experts.named_parameters():
                    if k == "up_proj.weight":
                        up_proj_weight = v
                    elif k == "up_proj.weight_scale":
                        up_proj_weight_scale = v
                    elif k == "down_proj.weight":
                        down_proj_weight = v
                    elif k == "down_proj.weight_scale":
                        down_proj_weight_scale = v
                if up_proj_weight is not None and up_proj_weight_scale is not None:
                    up_proj_weight_dq = dequantize_fp8_pb_wo_to_bf16(up_proj_weight, up_proj_weight_scale)
                    print(f"up_proj_weight_dq: {up_proj_weight_dq.shape=!r} {up_proj_weight_dq.dtype=!r} {up_proj_weight_dq.device=!r} {up_proj_weight_dq=!r}")
                else:
                    print(f"up_proj_weight: {up_proj_weight.shape=!r} {up_proj_weight.dtype=!r} {up_proj_weight.device=!r} {up_proj_weight=!r}")
                if down_proj_weight is not None and down_proj_weight_scale is not None:
                    down_proj_weight_dq = dequantize_fp8_pb_wo_to_bf16(down_proj_weight, down_proj_weight_scale)
                    print(f"down_proj_weight_dq: {down_proj_weight_dq.shape=!r} {down_proj_weight_dq.dtype=!r} {down_proj_weight_dq.device=!r} {down_proj_weight_dq=!r}")
                else:
                    print(f"down_proj_weight: {down_proj_weight.shape=!r} {down_proj_weight.dtype=!r} {down_proj_weight.device=!r} {down_proj_weight=!r}")
                print("="*100)

                print("="*100)
                print(f"shared_out: {shared_out.shape=!r} {shared_out.dtype=!r} {shared_out.device=!r} {shared_out=!r}")
                print(f"super(): {super()=!r}")
                print("="*100)

            fused_out = super().forward(
                hidden_states=hidden_states,
                router_logits=router_logits,
            )
        else:
            # went into here if set self.use_overlapped = True!!
            shared_out, fused_out = super().forward(
                hidden_states=hidden_states,
                router_logits=router_logits,
            )
            # ensure early TP reduction of shared expert outputs when required
            if (
                shared_out is not None
                and self.reduce_results
                and get_tensor_model_parallel_world_size() > 1
                and self.must_reduce_shared_expert_outputs()
            ):
                shared_out = tensor_model_parallel_all_reduce(shared_out)
        return shared_out, fused_out
