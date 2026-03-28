#!/bin/bash
# Chạy trên TEA
set -e

cd ~/lewm-lio && source .venv/bin/activate
git pull origin main

python src/eval.py \
    --config-path ../config/eval \
    --config-name nuscenes_nav \
    policy.checkpoint=outputs/checkpoints/best.ckpt

echo "Eval results: ~/lewm-lio/outputs/eval/"
