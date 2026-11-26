#!/usr/bin/env bash
# Environment used for the Week 9 run on a GB300 node (aarch64, driver CUDA 13.2).
# On x86 + CUDA 12.x the official `agentflow/setup.sh` is enough; on aarch64 its pinned
# GPU stack has no working wheels, so this script records every extra step that was run.
set -ex
export PATH=$HOME/.local/bin:$PATH
cd "$(dirname "$0")/../agentflow"

# 1. Official setup.sh, minus `sudo apt-get install jq` and the Ray restart.
uv venv -p 3.11 .venv && source .venv/bin/activate
(cd agentflow && uv pip install -r requirements.txt && uv pip install --no-deps -e .)
uv pip install -e .
uv pip install dashscope fire autogen-agentchat "autogen-ext[openai]" "litellm[proxy]" mcp openai-agents \
  langgraph "langchain[openai]" langchain-community langchain-text-splitters sqlparse nltk
uv pip install packaging ninja numpy pandas ipython ipykernel gdown wheel setuptools
uv pip install torch==2.7.0 torchvision==0.22.0 torchaudio==2.7.0 --index-url https://download.pytorch.org/whl/cu128
uv pip install transformers==4.53.3 verl==0.5.0
uv pip install -e ".[dev,agent]"
uv pip install yq

# 2. Pin packages that setup.sh now resolves to incompatible newer releases.
uv pip install transformers==4.53.3 "openai==1.75.0" "setuptools<80" "datasets==3.6.0" py-libnuma \
  "cupy-cuda12x<14" "sympy==1.13.3" "pyarrow<20" "mcp>=1.11,<2" faiss-cpu==1.12.0 matplotlib

# 3. CUDA 12.8 toolkit in $HOME (matches torch cu128) to build flash-attn and vLLM for sm_100
#    (sm_100 kernels run on the GB300's sm_103).
curl -fsSLo /tmp/cuda128.run https://developer.download.nvidia.com/compute/cuda/12.8.1/local_installers/cuda_12.8.1_570.124.06_linux_sbsa.run
sh /tmp/cuda128.run --silent --toolkit --installpath=$HOME/cuda-12.8 --no-man-page --override
export CUDA_HOME=$HOME/cuda-12.8 PATH=$HOME/cuda-12.8/bin:$PATH MAX_JOBS=96 NVCC_THREADS=2
FLASH_ATTN_CUDA_ARCHS=100 uv pip install flash-attn==2.8.1 --no-build-isolation

# 4. vLLM 0.9.2 from source (the PyPI aarch64 wheel lacks its CUDA kernels).
git clone --depth 1 -b v0.9.2 https://github.com/vllm-project/vllm.git /tmp/vllm-src
(cd /tmp/vllm-src && python use_existing_torch.py && uv pip install -r requirements/build.txt "setuptools<80" \
  && TORCH_CUDA_ARCH_LIST=10.0 VLLM_FA_CMAKE_GPU_ARCHES=100-real uv pip install --no-build-isolation --no-deps .)

# 5. Triton 3.3.0 cannot generate code on this host; 3.5.1 works with a CUDA >= 12.9 ptxas.
uv pip install triton==3.5.1

# 6. LoRA support for AgentFlow's async rollout server.
patch -d .venv/lib/python3.11/site-packages -p1 < ../patches/verl-0.5.0-async-lora-merge.patch
