#!/bin/bash
# Phase 2 - seeds and ablations, planned after phase 1 (results/summary.csv):
#   Model 2 reaches 89-90 % validation accuracy with every learning rate from 1e-5 to 5e-5
#   (1e-5 and 3e-5 tie at 90.2 %). Model 1 reaches 66-67 % with 1e-5 to 5e-5 and stays at chance with 1e-4.
#
# A. Seeds 1, 2, 3 in addition to 42: how large is the seed noise, and which learning rate wins the tie?
#      model 2 with 1e-5 and with 3e-5, model 1 with 2e-5 (best validation accuracy in phase 1)
#
# B. Ablations with seed 42. Each run changes one factor against its reference run
#    (model 2: pre-lr1e-05-s42, model 1: rand-lr2e-05-ep30-s42).
#      1. order of the segments: o1_o2_h instead of o1_h_o2
#      2. pooling: mean of all tokens instead of the first token
#      3. hypothesis only: how much can be decided without the observations?
#      4. training examples: 1000 and 4000 instead of 16801 - how much data does pretraining save?
#      5. model 2 only: classifier learning rate 1e-3 (the classifier is randomly initialized)
#      6. model 2 only: 10 instead of 5 epochs (with 1e-5 the best epoch was the last one)
#
# All decisions use the validation accuracy. The test split is only reported.
#
#   mkdir -p logs && nohup bash scripts/run_phase2.sh > logs/phase2.log 2>&1 &

export WANDB_ENTITY=${WANDB_ENTITY:-gabriel-gjonaj-hochschule-luzern}
PRE="python train.py --pretrained true"
RAND="python train.py --pretrained false --lr 2e-5 --max_epochs 30 --patience 5"

# A. seeds
for seed in 1 2 3; do
    $PRE --lr 1e-5 --seed $seed --tags phase2,seeds
    $PRE --lr 3e-5 --seed $seed --tags phase2,seeds
    $RAND --seed $seed --tags phase2,seeds
done

# B. ablations
for option in "--seq_format o1_o2_h" "--pooling mean" "--seq_format h_only" "--train_size 1000" "--train_size 4000"; do
    $PRE --lr 1e-5 $option --tags phase2,ablation
    $RAND $option --tags phase2,ablation
done
$PRE --lr 1e-5 --classifier_lr 1e-3 --tags phase2,ablation
$PRE --lr 1e-5 --max_epochs 10 --patience 3 --tags phase2,ablation
