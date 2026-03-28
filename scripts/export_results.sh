#!/bin/bash
# Chạy trên TEA — Pack results để copy về CONT
set -e

cd ~/lewm-lio
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
EXPORT_DIR="outputs/export_${TIMESTAMP}"
mkdir -p "$EXPORT_DIR"

cp -r outputs/eval/        "$EXPORT_DIR/eval/"        2>/dev/null || true
cp -r outputs/degeneracy/  "$EXPORT_DIR/degeneracy/"  2>/dev/null || true
cp -r outputs/logs/        "$EXPORT_DIR/logs/"        2>/dev/null || true
cp outputs/checkpoints/best.ckpt "$EXPORT_DIR/"       2>/dev/null || true

tar -czf "outputs/results_${TIMESTAMP}.tar.gz" -C outputs "export_${TIMESTAMP}"

echo "Exported: outputs/results_${TIMESTAMP}.tar.gz ($(du -h outputs/results_${TIMESTAMP}.tar.gz | cut -f1))"
echo ""
echo "Copy về CONT:"
echo "  scp $(whoami)@$(hostname -I | awk '{print $1}'):~/lewm-lio/outputs/results_${TIMESTAMP}.tar.gz ~/lewm-lio/outputs/"
