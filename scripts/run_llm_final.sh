#!/bin/bash
# Model 3, step 3 - final run: the selected model and prompt once on the test split.
#
# Result of step 2 (whole validation split, 1000 examples, accuracy in %):
#   Qwen3 235B   zero-shot 95.3 | few-shot 4: 95.8 | few-shot 8: 96.1 | step by step 95.8 | zero-shot, hypotheses swapped 95.7
#   gpt-oss 120B zero-shot 95.5
# Invalid answers: few-shot 8: 2 ("Neither middle is plausible"), step by step: 10 (9 of them reached the limit of 1024 tokens).
# Paired comparison with zero-shot (exact McNemar test): no difference is significant (p = 0.24 to 0.89).
# Swapping the hypotheses changes 4.6 % of the predictions and not the accuracy.
# Choice by the highest validation accuracy: Qwen3 235B with 8 solved examples. The margin over zero-shot
# (0.8 points) is within the noise; step by step costs 150 times more output tokens without a gain.
#
# The run continues the folder of step 2, so the validation answers come from the cache.
# Train: 1000 random training examples that are not among the 8 solved examples.
# Test: all 1532 examples, used here for the first time.
#
#   mkdir -p logs && nohup bash scripts/run_llm_final.sh > logs/llm_final.log 2>&1 &

export WANDB_ENTITY=${WANDB_ENTITY:-gabriel-gjonaj-hochschule-luzern}
python llm.py --model Qwen/Qwen3-235B-A22B-Instruct-2507 --provider deepinfra --prompt few --shots 8 \
    --splits train,validation,test --run_name llm-qwen3-235b-few8 --tags llm,final
