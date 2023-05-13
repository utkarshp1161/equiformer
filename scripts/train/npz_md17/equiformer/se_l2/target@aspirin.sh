#!/bin/bash

# # Loading the required module
# source /etc/profile
# module load anaconda/2021a

# utk_eq
export PYTHONNOUSERSITE=True    # prevent using packages from base


python main_npz_md17.py \
    --output-dir '/home/sire/phd/srz228573/equiformer/data_sl/fone_output/equiformer/npz_md17/se_l2/aspirin' \
    --model-name 'graph_attention_transformer_nonlinear_exp_l2_md17' \
    --input-irreps '64x0e' \
    --target 'aspirin' \
    --data-path '/home/sire/phd/srz228573/equiformer/data_sl/fone_dataset/mdsim_data/md17/aspirin/10k/' \
    --epochs 1500 \
    --lr 5e-4 \
    --batch-size 8 \
    --weight-decay 1e-6 \
    --num-basis 32 \
    --energy-weight 1 \
    --force-weight 80
