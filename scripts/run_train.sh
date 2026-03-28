#!/bin/bash
# Chạy trên TEA
set -e

cd ~/lewm-lio && source .venv/bin/activate
git pull origin main

echo "Training LeWM-LiDAR on $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"

python src/train.py \
    --config-path ../config/train \
    --config-name lewm_lidar \
    training.device=cuda \
    training.batch_size=64

echo "Done. Checkpoints: ~/lewm-lio/outputs/checkpoints/"
