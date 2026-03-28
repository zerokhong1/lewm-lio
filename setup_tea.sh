#!/bin/bash
# ================================================================
# setup_tea.sh — Chạy trên máy TEA (nghiên cứu)
# Không cần đăng nhập bất kỳ dịch vụ nào
# ================================================================
set -e

echo "========================================="
echo "  SETUP TEA — LeWM-LIO Project"
echo "========================================="

# —— Config ——
PROJECT_DIR="$HOME/lewm-lio"
DATA_DIR="$HOME/lewm-lio-data"
STABLEWM_HOME="$DATA_DIR/stable-wm"
PYTHON_VERSION="3.10"

# —— 1. Clone project (public repo, không cần auth) ——
echo "[1/6] Cloning project repo..."
if [ ! -d "$PROJECT_DIR" ]; then
    git clone https://github.com/<YOUR_USERNAME>/lewm-lio.git "$PROJECT_DIR"
else
    cd "$PROJECT_DIR" && git pull origin main
fi

# —— 2. Clone upstream dependencies ——
echo "[2/6] Cloning upstream repos..."
mkdir -p "$PROJECT_DIR/upstream"
cd "$PROJECT_DIR/upstream"

[ ! -d "le-wm" ]     && git clone https://github.com/lucas-maes/le-wm.git
[ ! -d "adljepa" ]   && git clone https://github.com/HaoranZhuExplorer/adljepa.git
[ ! -d "lejepa" ]    && git clone https://github.com/galilai-group/lejepa.git
[ ! -d "dino_wm" ]   && git clone https://github.com/gaoyuezhou/dino_wm.git
[ ! -d "OpenPCDet" ] && git clone https://github.com/open-mmlab/OpenPCDet.git

# —— 3. Setup Python env ——
echo "[3/6] Setting up Python environment..."
cd "$PROJECT_DIR"

if ! command -v uv &> /dev/null; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
fi

uv venv --python=$PYTHON_VERSION
source .venv/bin/activate

uv pip install -r requirements.txt

# —— 4. Cài stable-worldmodel ——
echo "[4/6] Installing stable-worldmodel..."
uv pip install "stable-worldmodel[train]"

# —— 5. Setup env vars ——
echo "[5/6] Setting environment variables..."
cat >> "$PROJECT_DIR/.venv/bin/activate" << ENVEOF

# —— LeWM-LIO env vars ——
export STABLEWM_HOME="$STABLEWM_HOME"
export LEWM_DATA_DIR="$DATA_DIR"
export LEWM_PROJECT_DIR="$PROJECT_DIR"
export PYTHONPATH="$PROJECT_DIR/src:\$PYTHONPATH"

# Disable WandB (TEA không cần internet auth)
export WANDB_MODE=offline

# CUDA
export CUDA_VISIBLE_DEVICES=0
ENVEOF

source .venv/bin/activate

# —— 6. Tạo thư mục ——
echo "[6/6] Creating directories..."
mkdir -p "$DATA_DIR"/{stable-wm,processed,results,checkpoints}
mkdir -p "$PROJECT_DIR"/outputs

echo "========================================="
echo "  SETUP COMPLETE"
echo ""
echo "  Activate env:  source $PROJECT_DIR/.venv/bin/activate"
echo "  Data dir:      $DATA_DIR"
echo "  Project dir:   $PROJECT_DIR"
echo ""
echo "  TIẾP THEO: Copy data files vào $DATA_DIR/"
echo "========================================="
