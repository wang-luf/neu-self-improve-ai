# Source before running vLLM, training or evaluation on the GB300 node.
export TRITON_PTXAS_PATH=/usr/local/cuda-13.1/bin/ptxas          # Triton 3.5.1 needs a ptxas that knows sm_103a
export TRITON_PTXAS_BLACKWELL_PATH=/usr/local/cuda-13.1/bin/ptxas
export TORCHDYNAMO_DISABLE=1   # torch 2.7's inductor is incompatible with Triton 3.5.1: run torch.compile regions eagerly
export VLLM_USE_V1=1           # as set by agentflow/scripts/restart_ray.sh
export JUDGE_BASE_URL=http://localhost:8000/v1                # local judge instead of gpt-4o
export WIKI_RETRIEVER_URL=http://127.0.0.1:8002/retrieve      # local wiki-18 search instead of Google/Wikipedia APIs
