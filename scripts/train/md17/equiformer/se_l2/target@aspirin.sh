#!/bin/bash

# # Loading the required module
# source /etc/profile
# module load anaconda/2021a

# utk_eq
export PYTHONNOUSERSITE=True    # prevent using packages from base


python main_md17.py \
    --output-dir '/home/sire/phd/srz228573/benchmarking_datasets/equiformer_output/se_l2/target@aspirin/lr@5e-4_wd@1e-6_epochs@1500_w-f2e@80_dropout@0.0_exp@32_l2mae-loss' \
    --model-name 'graph_attention_transformer_nonlinear_exp_l2_md17' \
    --input-irreps '64x0e' \
    --target 'aspirin' \
    --data-path '/home/sire/phd/srz228573/benchmarking_datasets/equiformer_data/md17' \
    --epochs 1500 \
    --lr 5e-4 \
    --batch-size 8 \
    --weight-decay 1e-6 \
    --num-basis 32 \
    --energy-weight 1 \
    --force-weight 80
