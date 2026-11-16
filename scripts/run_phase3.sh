#!/bin/bash
# Phase 3 - seeds for the ablations that carry a claim, planned after the evaluation (notebooks/analysis.ipynb).
#
# In phase 2 every ablation ran with seed 42 only. The standard deviation over the seeds of the final
# configurations is 1.0 points for model 1 and 0.3 points for model 2 (validation accuracy), so one run
# is not enough for a claim. Three ablations carry a claim of the project:
#
# A. Hypothesis only, both models: is model 1 really as good without the observations?
#      Seed 42: 67.8 % without and 67.4 % with the observations - the difference is smaller than the seed noise.
# B. 1000 and 4000 training examples, both models: how much data does pretraining save?
#      The subset is drawn with the seed, so the seeds also vary the training examples.
# C. Segment order o1_o2_h, model 2: is the order o1_h_o2 really better?
#      Seed 42: 88.2 % instead of 90.2 %. This comparison backs the choice of the input format.
#      Model 2 only: for model 1 the difference (66.1 % instead of 67.4 %) is inside the seed noise.
#
# Seeds 1, 2, 3 in addition to 42 - the same seeds as for the final configurations. 21 runs, about 3 hours.
# The other ablations (pooling, classifier learning rate, number of epochs) stay at one seed: their
# differences are inside the seed noise and they do not change a decision.
# All comparisons use the validation accuracy. The test split is only reported.
#
#   mkdir -p logs && nohup bash scripts/run_phase3.sh > logs/phase3.log 2>&1 &

export WANDB_ENTITY=${WANDB_ENTITY:-gabriel-gjonaj-hochschule-luzern}
PRE="python train.py --pretrained true --lr 1e-5"
RAND="python train.py --pretrained false --lr 2e-5 --max_epochs 30 --patience 5"

# A. hypothesis only
for seed in 1 2 3; do
    $PRE --seq_format h_only --seed $seed --tags phase3,ablation
    $RAND --seq_format h_only --seed $seed --tags phase3,ablation
done

# B. training set size
for seed in 1 2 3; do
    for size in 1000 4000; do
        $PRE --train_size $size --seed $seed --tags phase3,ablation
        $RAND --train_size $size --seed $seed --tags phase3,ablation
    done
done

# C. segment order, model 2
for seed in 1 2 3; do
    $PRE --seq_format o1_o2_h --seed $seed --tags phase3,ablation
done
