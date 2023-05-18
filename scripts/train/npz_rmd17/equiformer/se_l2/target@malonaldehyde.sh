# #!/bin/bash

# # Loading the required module
# source /etc/profile
# module load anaconda/2021a

export PYTHONNOUSERSITE=True    # prevent using packages from base
source activate th102_cu113_tgconda

python main_md17.py \
    --output-dir '/home/sire/phd/srz228573/equiformer/data_sl/fone_output/equiformer/npz_md17/se_l2/' \
    --model-name 'graph_attention_transformer_nonlinear_exp_l2_md17' \
    --input-irreps '64x0e' \
    --target 'malonaldehyde' \
    --data-path '/home/sire/phd/srz228573/benchmarking_datasets/fone_dataset/mdsim_data/md17/malonaldehyde/10k' \
    --epochs 1500 \
    --lr 5e-4 \
    --batch-size 8 \
    --weight-decay 1e-6 \
    --num-basis 32 \
    --energy-weight 1 \
    --force-weight 100
