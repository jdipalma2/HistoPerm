#!/bin/bash

# VICReg ablations
# Baseline
python -u run.py --mode="vicreg" --dataset_name="dhmc_cd" --shuffle_percentage=0
# Run the linear mode
python -u run.py --mode="linear_vicreg" --dataset_name="dhmc_cd" --shuffle_percentage=0 --lr=0.2 --max_epochs=80 --warmup_epochs=0 --weight_decay=0 --ckpt_file="/workspace/jdipalma/BYOL/src_cd_data_aug_ablations_with_vicreg_and_baseline/logs/vicreg/lightning_logs/version_0/checkpoints/epoch=49.ckpt"


# VICReg+HistoPerm ablations
# Baseline
python -u run.py --mode="vicreghp" --dataset_name="dhmc_cd" --shuffle_percentage=0.5
# Run the linear mode
python -u run.py --mode="linear_vicreghp" --dataset_name="dhmc_cd" --shuffle_percentage=0.5 --lr=0.2 --max_epochs=80 --warmup_epochs=0 --weight_decay=0 --ckpt_file="/workspace/jdipalma/BYOL/src_cd_data_aug_ablations_with_vicreg_and_baseline/logs/vicreghp/lightning_logs/version_0/checkpoints/epoch=49.ckpt"

