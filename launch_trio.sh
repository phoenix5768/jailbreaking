#!/bin/bash
# 1. Activate Environment
source /scratch-scc/projects/ag_gipp/u24012/jailbreak_project/venv/bin/activate

# 2. Define Shared Arguments
# --tensor-parallel-size 1: Fits on 1 GPU
# --gpu-memory-utilization 0.3: STRICT LIMIT of 30% VRAM per model
COMMON_ARGS="--tensor-parallel-size 1 --gpu-memory-utilization 0.3 --max-model-len 8192 --trust-remote-code --dtype bfloat16"


JB_MODEL = "/scratch-scc/projects/ag_gipp/u24012/models/DeepSeek-Coder-6.7B-Instruct"
VIC_MODEL = "/scratch-scc/projects/ag_gipp/u24012/models/Qwen2.5-7B-Instruct"
JUDGE_MODEL = "/scratch-scc/projects/ag_gipp/u24012/models/Meta-Llama-Guard-2-8B"

# 3. Launch Attacker (DeepSeek) -> Port 8000
echo "Starting Attacker..."
python3 -m vllm.entrypoints.openai.api_server \
    --model $JB_MODEL \
    --port 8000 \
    --served-model-name attacker \
    $COMMON_ARGS > attacker.log 2>&1 &

# 4. Launch Victim (Qwen2.5) -> Port 8001
echo "Starting Victim..."
python3 -m vllm.entrypoints.openai.api_server \
    --model $VIC_MODEL \
    --port 8001 \
    --served-model-name victim \
    $COMMON_ARGS > victim.log 2>&1 &

# 5. Launch Judge (Llama Guard) -> Port 8002
# Llama Guard might need a smaller max_model_len to save memory
echo "Starting Judge..."
python3 -m vllm.entrypoints.openai.api_server \
    --model $JUDGE_MODEL \
    --port 8002 \
    --served-model-name judge \
    $COMMON_ARGS > judge.log 2>&1 &

# 6. Wait
echo "Models are initializing. Check log files (attacker.log, etc.) for 'Application startup complete'."
wait