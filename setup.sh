#!/usr/bin/env bash
# setup.sh — create and activate the norms-k-calibration virtual environment.
#
# MUST be run with source so the venv activation persists in your shell:
#
#   source setup.sh              # first-time: create venv + install deps
#   source setup.sh              # subsequent: activate only (skips install)
#   source setup.sh --reinstall  # force-reinstall all packages
#
# Optional env vars:
#   VENV_DIR   path to the venv directory  (default: ./venv)
#   PYTHON     python binary to use        (default: python3)

# Guard: must be sourced, not executed
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    echo "ERROR: run with 'source setup.sh', not 'bash setup.sh'."
    echo "       A sub-process cannot activate the venv in your current shell."
    exit 1
fi

VENV_DIR="${VENV_DIR:-venv}"
PYTHON="${PYTHON:-python3}"
REINSTALL=false

for _arg in "$@"; do
    [[ "$_arg" == "--reinstall" ]] && REINSTALL=true
done

# ------------------------------------------------------------------
# 1. Create venv if missing
# ------------------------------------------------------------------
if [[ ! -d "$VENV_DIR" ]]; then
    echo "[setup] Creating venv at ./$VENV_DIR ..."
    "$PYTHON" -m venv "$VENV_DIR"
fi

# ------------------------------------------------------------------
# 2. Activate
# ------------------------------------------------------------------
source "$VENV_DIR/bin/activate"
echo "[setup] Activated: $(which python)  ($(python --version))"

# ------------------------------------------------------------------
# 3. Skip install if venv is already populated and --reinstall not set
# ------------------------------------------------------------------
if [[ "$REINSTALL" == false ]] && python -c "import transformers" &>/dev/null 2>&1; then
    echo "[setup] Packages already installed. Use 'source setup.sh --reinstall' to force."
    return 0
fi

# ------------------------------------------------------------------
# 4. Upgrade pip + wheel
# ------------------------------------------------------------------
echo "[setup] Upgrading pip ..."
pip install --quiet --upgrade pip wheel

# ------------------------------------------------------------------
# 5. Install PyTorch with the correct CUDA index URL
#    Auto-detects CUDA version from nvcc; falls back to CPU.
# ------------------------------------------------------------------
if command -v nvcc &>/dev/null; then
    _CUDA_VER=$(nvcc --version | grep -oP 'release \K[0-9]+\.[0-9]+' | head -1)
    # "12.1" → "cu121",  "12.4" → "cu124"
    _CUDA_TAG="cu$(echo "$_CUDA_VER" | tr -d '.')"
    echo "[setup] Detected CUDA $_CUDA_VER — installing torch for $_CUDA_TAG ..."
    pip install --quiet \
        "torch>=2.4.0" "torchvision>=0.19.0" \
        --index-url "https://download.pytorch.org/whl/${_CUDA_TAG}"
else
    echo "[setup] nvcc not found — installing CPU-only torch (no GPU acceleration)."
    pip install --quiet "torch>=2.4.0" "torchvision>=0.19.0"
fi

# ------------------------------------------------------------------
# 6. Install remaining requirements
#    torch is already satisfied above; pip will skip re-downloading it.
# ------------------------------------------------------------------
echo "[setup] Installing project requirements ..."
pip install --quiet -r requirements.txt

# ------------------------------------------------------------------
# 7. Install package in editable mode
# ------------------------------------------------------------------
echo "[setup] Installing norms-k-calibration (editable) ..."
pip install --quiet -e . --no-deps

# ------------------------------------------------------------------
# Done
# ------------------------------------------------------------------
echo ""
echo "[setup] Environment ready. Example run:"
echo "   python experiments/10_downstream_behavioral_eval.py \\"
echo "       --model llama_8b --behavior all --dtype bf16 --compile"
