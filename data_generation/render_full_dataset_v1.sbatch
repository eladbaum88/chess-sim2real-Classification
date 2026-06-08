#!/bin/bash
#SBATCH --partition=course
#SBATCH --qos=course
#SBATCH --gres=gpu:rtx_3090:1
#SBATCH --time=06:00:00
#SBATCH --job-name=ds_v1_full
#SBATCH --output=/home/eladbaum/chess_project/Project2_3/dataset_v1_full.out
#SBATCH --error=/home/eladbaum/chess_project/Project2_3/dataset_v1_full.err

set -e
date
hostname
nvidia-smi -L

cd /home/eladbaum/chess_project/Project2_3

# --resume preserves the 12 smoke-test images already present
python build_dataset_v1.py \
    --from-v1-labels \
    --runs-per-fen 4 \
    --outer-padding 0.03 \
    --piece-margin 0.2 \
    --resume

date
echo "--- final image count ---"
ls dataset_v1/images/ | wc -l
echo "--- final labels.csv row count (incl header) ---"
wc -l dataset_v1/labels.csv
