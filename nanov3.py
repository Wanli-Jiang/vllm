""" CMDs and instructions:

(If we want to use DeepGEMM kernel, we need to install DeepGemm firstly):

```bash
# Submodule must be cloned
git clone --recursive git@github.com:deepseek-ai/DeepGEMM.git
cd DeepGEMM
cat develop.sh
./develop.sh
cat install.sh
./install.sh
pip show deepgemm
```

* For H200, the following CMDs to run the model:

VLLM_USE_FLASHINFER_MOE_FP8=0  VLLM_USE_DEEP_GEMM=0 VLLM_MOE_USE_DEEP_GEMM=0 python nanov3.py
- it will use MOE_triton kernel, can run but no output (generated token ids are all 0)

VLLM_USE_FLASHINFER_MOE_FP8=1  VLLM_USE_DEEP_GEMM=0 VLLM_MOE_USE_DEEP_GEMM=0 python nanov3.py
- it will use MOE_CUTLASS kernel, can run but no output (generated token ids are all 0)

VLLM_USE_FLASHINFER_MOE_FP8=0  VLLM_USE_DEEP_GEMM=1 VLLM_MOE_USE_DEEP_GEMM=1 python nanov3.py
- it will use MOE_DeepGEMM kernel, can run but no output (generated token ids are all 0)


* For B200, the following CMDs to run the model:

VLLM_USE_FLASHINFER_MOE_FP8=0  VLLM_USE_DEEP_GEMM=0 VLLM_MOE_USE_DEEP_GEMM=0 python nanov3.py
- it will use MOE_triton kernel, can run but no output (generated token ids are all 0)

VLLM_USE_FLASHINFER_MOE_FP8=1  VLLM_USE_DEEP_GEMM=0 VLLM_MOE_USE_DEEP_GEMM=0 python nanov3.py
- it will use MOE_CUTLASS kernel, it will fail due to incompatible with B200.

VLLM_USE_FLASHINFER_MOE_FP8=0  VLLM_USE_DEEP_GEMM=1 VLLM_MOE_USE_DEEP_GEMM=1 python nanov3.py
- it will use MOE_DeepGEMM kernel, can run but no output (generated token ids are all 0)
"""

from vllm import LLM, SamplingParams

"""
The following models are in computelab cluster, see `/home/scratch.williamj_coreai/models/`.
"""
# llm = LLM(model="/code/wj-models/nano-v3-row73-1125", trust_remote_code=True)
# llm = LLM(model="/code/wj-models/NVIDIA-Nemotron-Nano-3-30B-A3.5B-config5-reasoning-calib-seq-len-8K-FP8-KVFP8_HF", trust_remote_code=True)
# llm = LLM(model="/code/wj-models/NVIDIA-Nemotron-Nano-3-30B-A3.5B-config1-reasoning-calib-seq-len-8K-NVFP4_HF", trust_remote_code=True)
# llm = LLM(model="/code/wj-models/NVIDIA-Nemotron-Nano-31B-A3-v3_nvfp4", trust_remote_code=True)

# llm = LLM(model="/code/wj-models/Nemotron-Nano-3-30B-A3.5B-dev-1024_fp8_pb_wo", trust_remote_code=True)
# llm = LLM(model="/code/wj-models/Nemotron-Nano-3-30B-A3.5B-dev-1024-wj-no-moe", trust_remote_code=True)
# llm = LLM(model="/code/wj-models/Nemotron-Nano-3-30B-A3.5B-dev-1024-wj-default", trust_remote_code=True, enforce_eager=True)
# llm = LLM(model="/code/wj-models/Nemotron-Nano-3-30B-A3.5B-dev-1024", trust_remote_code=True, enforce_eager=True)

llm = LLM(model="/code/wj-models/Nemotron-Nano-3-30B-A3.5B-dev-1024-fp8_pb_wo_only_moe", trust_remote_code=True, enforce_eager=True)

# llm = LLM(model="/code/wj-models/Qwen3-30B-A3B-FP8", trust_remote_code=True)

# llm = LLM(model="/code/wj-models/Qwen3-Next-80B-A3B-Instruct-FP8", trust_remote_code=True)
# llm = LLM(model="/code/wj-models/DeepSeek-V3-Lite-fp8", trust_remote_code=True)


sampling_params = SamplingParams(
  max_tokens=32,
  temperature=0.0,
)

text = [
    "Hello, my name is",
    # "The capital of France is",
    # "The future of AI is",
]

outputs = llm.generate(text, sampling_params)

for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
