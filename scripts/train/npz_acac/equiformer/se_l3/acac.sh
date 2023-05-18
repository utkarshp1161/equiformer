#!/bin/bash

# Loading the required module
# source /etc/profile
# module load anaconda/2021a

export PYTHONNOUSERSITE=True    # prevent using packages from base
# source activate th102_cu113_tgconda

python main_npz_acac.py  \
    --output-dir '/home/sire/phd/srz228573/equiformer/data_sl/fone_output/equiformer/npz_acac_300K/se_l3/' \
    --model-name 'graph_attention_transformer_nonlinear_exp_l3_md17' \
    --input-irreps '64x0e' \
    --target 'lips' \
    --data-path '/home/sire/phd/srz228573/benchmarking_datasets/mace_data/BOTNet-datasets/dataset_acac/nequip_format/300K/' \
    --epochs 2000 \
    --lr 2e-4 \
    --batch-size 5 \
    --eval-batch-size 16 \
    --weight-decay 1e-6 \
    --num-basis 32 \
    --energy-weight 1 \
    --force-weight 100
