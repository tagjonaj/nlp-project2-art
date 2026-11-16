#!/bin/bash
# Model 3, step 2 - which prompt? Whole validation split (1000 examples).
#
# Result of step 1 (zero-shot, 200 validation examples, accuracy in %):
#   Llama 3.3 70B 95.5 | DeepSeek V4 Pro 96.0 | gpt-oss 120B 98.0 | Qwen3 235B 98.0 | GLM 5.2 98.5 (1 invalid answer)
# With 200 examples the standard error is 1.0 to 1.5 points: the pilot does not single out one model.
# Choice: Qwen3 235B Instruct. It is in the best group and answers directly (2 output tokens per example),
# so the prompt formats are compared without reasoning that the prompt did not ask for.
# gpt-oss 120B (best group, no invalid answer) is kept as the model that reasons before every answer.
#
# A. Prompt format with Qwen3 235B: zero-shot, few-shot with 4 and with 8 solved examples, step by step
# B. Position of the hypotheses: zero-shot with the two hypotheses swapped - is there a position bias?
# C. Reasoning model: gpt-oss 120B, zero-shot
#
# All decisions use the validation accuracy. The test split is not used.
#
#   mkdir -p logs && nohup bash scripts/run_llm_prompts.sh > logs/llm_prompts.log 2>&1 &

export WANDB_ENTITY=${WANDB_ENTITY:-gabriel-gjonaj-hochschule-luzern}
QWEN="python llm.py --model Qwen/Qwen3-235B-A22B-Instruct-2507 --provider deepinfra --splits validation --tags llm,prompts"

# A. prompt format
$QWEN --prompt zero --run_name llm-qwen3-235b-zero
$QWEN --prompt few --shots 4 --run_name llm-qwen3-235b-few4
$QWEN --prompt few --shots 8 --run_name llm-qwen3-235b-few8
$QWEN --prompt cot --max_tokens 1024 --run_name llm-qwen3-235b-cot

# B. position of the hypotheses
$QWEN --prompt zero --swap true --run_name llm-qwen3-235b-zero-swap

# C. reasoning model
python llm.py --model openai/gpt-oss-120b --provider deepinfra --max_tokens 2048 --splits validation --prompt zero \
    --tags llm,prompts --run_name llm-gpt-oss-120b-zero
