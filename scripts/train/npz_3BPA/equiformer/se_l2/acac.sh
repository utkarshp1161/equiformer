#!/bin/bash

# # Loading the required module
# source /etc/profile
# module load anaconda/2021a

# utk_eq
export PYTHONNOUSERSITE=True    # prevent using packages from base


python main_npz_acac.py \
    --output-dir '/home/sire/phd/srz228573/equiformer/data_sl/fone_output/equiformer/npz_lips/se_l2/' \
    --model-name 'graph_attention_transformer_nonlinear_exp_l2_md17' \
    --input-irreps '64x0e' \
    --target 'lips' \
    --data-path '/home/sire/phd/srz228573/benchmarking_datasets/mace_data/BOTNet-datasets/dataset_acac/nequip_format/300K/' \
    --epochs 1500 \
    --lr 5e-4 \
    --batch-size 8 \
    --weight-decay 1e-6 \
    --num-basis 32 \
    --energy-weight 1 \
    --force-weight 80
