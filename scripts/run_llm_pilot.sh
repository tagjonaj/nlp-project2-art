#!/bin/bash
# Model 3, step 1 - which LLM? Zero-shot prompt on 200 random validation examples (same examples for every model).
#
# Candidates: open-weight models from 70 to 1600 billion parameters, each served by one fixed inference provider.
#   Llama 3.3 70B          comparable to the published zero-shot reference (Llama 3.1 70B, 86.2 %)
#   gpt-oss 120B           reasoning model
#   Qwen3 235B Instruct    no reasoning
#   GLM 5.2 (753B) and DeepSeek V4 Pro (1599B)   largest models available
# Models that reason before they answer get 2048 instead of 8 tokens, otherwise the answer is cut off.
#
# With 200 examples and 90 % accuracy the standard error is 2 points: this step only removes clearly weaker
# models. The remaining candidates and the prompt formats are compared on the whole validation split.
# The test split is not used.
#
#   bash scripts/run_llm_pilot.sh 5                                                # check of the API: 5 examples, no tracking
#   mkdir -p logs && nohup bash scripts/run_llm_pilot.sh > logs/llm_pilot.log 2>&1 &

export WANDB_ENTITY=${WANDB_ENTITY:-gabriel-gjonaj-hochschule-luzern}
LIMIT=${1:-200}
OPTIONS="--prompt zero --splits validation --limit $LIMIT --tags llm,pilot"
if [ "$LIMIT" -lt 200 ]; then
    OPTIONS="$OPTIONS --wandb_mode disabled --output_dir /tmp/llm_check"
fi

python llm.py --model meta-llama/Llama-3.3-70B-Instruct --provider novita $OPTIONS
python llm.py --model openai/gpt-oss-120b --provider deepinfra --max_tokens 2048 $OPTIONS
python llm.py --model Qwen/Qwen3-235B-A22B-Instruct-2507 --provider deepinfra $OPTIONS
python llm.py --model zai-org/GLM-5.2 --provider deepinfra --max_tokens 2048 $OPTIONS
python llm.py --model deepseek-ai/DeepSeek-V4-Pro --provider deepinfra --max_tokens 2048 $OPTIONS
