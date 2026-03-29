#!/bin/bash
# ─────────────────────────────────────────
# scripts/run_ablations.sh
# Chạy trên TEA. Mỗi ablation = 1 training run.
# ─────────────────────────────────────────
set -e

cd ~/lewm-lio
conda activate lewm

DATA="data/processed/nuscenes_mini_bev.h5"
STEPS=10000
COMMON="--data_path $DATA --max_steps $STEPS --seq_length 16 --bev_size 256 --print_every 200 --save_every 5000 --num_workers 4 --device cuda"

echo "================================="
echo "  Ablation Suite"
echo "================================="

# ── Ablation 1: SIGReg lambda ──
echo ""
echo "[1/3] Ablation: SIGReg lambda"
for LAMBDA in 0.0 0.1 1.0 10.0; do
    echo "  lambda=$LAMBDA"
    python src/train.py $COMMON \
        --batch_size 16 \
        --sigreg_lambda $LAMBDA \
        --log_dir outputs/ablation/lambda_${LAMBDA}/logs \
        --ckpt_dir outputs/ablation/lambda_${LAMBDA}/ckpts
done

# ── Ablation 2: Latent dimension ──
echo ""
echo "[2/3] Ablation: Latent dimension"
for DIM in 64 128 192 384; do
    echo "  latent_dim=$DIM"
    python src/train.py $COMMON \
        --batch_size 16 \
        --latent_dim $DIM \
        --sigreg_lambda 1.0 \
        --log_dir outputs/ablation/dim_${DIM}/logs \
        --ckpt_dir outputs/ablation/dim_${DIM}/ckpts
done

# ── Ablation 3: BEV resolution ──
echo ""
echo "[3/3] Ablation: BEV resolution"
for RES in 64 128 256; do
    echo "  bev_size=$RES"
    python src/train.py $COMMON \
        --batch_size 16 \
        --bev_size $RES \
        --sigreg_lambda 1.0 \
        --log_dir outputs/ablation/bev_${RES}/logs \
        --ckpt_dir outputs/ablation/bev_${RES}/ckpts
done

echo ""
echo "================================="
echo "  All ablations complete"
echo "  Results in: outputs/ablation/"
echo "================================="
