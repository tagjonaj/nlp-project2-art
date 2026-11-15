#!/bin/bash
# Phase 1 - learning rate search, one seed per configuration.
#
# Question: which learning rate does each model need?
#   Model 2 (pretrained):  1e-5, 2e-5, 3e-5, 5e-5 - the range recommended for finetuning BERT and RoBERTa.
#   Model 1 (random init): 1e-5, 2e-5, 5e-5, 1e-4 - the pilot run with 1e-4 fell back to chance after the
#                          first epoch, so the search goes down from there.
# The two pilot runs (model 2 with 2e-5, model 1 with 1e-4) have exactly these settings and are part of
# the search. This script runs the six remaining configurations.
# Selection: best validation accuracy. The test split is not used for any decision.
#
#   mkdir -p logs && nohup bash scripts/run_phase1.sh > logs/phase1.log 2>&1 &

export WANDB_ENTITY=${WANDB_ENTITY:-gabriel-gjonaj-hochschule-luzern}

for lr in 1e-5 3e-5 5e-5; do
    python train.py --pretrained true --lr $lr --tags phase1
done

for lr in 1e-5 2e-5 5e-5; do
    python train.py --pretrained false --lr $lr --max_epochs 30 --patience 5 --tags phase1
done
