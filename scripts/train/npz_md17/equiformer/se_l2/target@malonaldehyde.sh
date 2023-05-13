# #!/bin/bash

# # Loading the required module
# source /etc/profile
# module load anaconda/2021a

export PYTHONNOUSERSITE=True    # prevent using packages from base
source activate th102_cu113_tgconda

python main_md17.py \
    --output-dir '/home/sire/phd/srz228573/benchmarking_datasets/equiformer_output/se_l2/target@malonaldehyde/lr@5e-4_wd@1e-6_epochs@1500_w-f2e@100_dropout@0.0_exp@32_l2mae-loss' \
    --model-name 'graph_attention_transformer_nonlinear_exp_l2_md17' \
    --input-irreps '64x0e' \
    --target 'malonaldehyde' \
    --data-path 'datasets/md17' \
    --epochs 1500 \
    --lr 5e-4 \
    --batch-size 8 \
    --weight-decay 1e-6 \
    --num-basis 32 \
    --energy-weight 1 \
    --force-weight 100