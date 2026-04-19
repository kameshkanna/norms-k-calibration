#!/usr/bin/env bash
# Run all experiments in sequence. Each experiment covers all models and behaviours internally.
# Usage:
#   bash run_all.sh              # run everything from exp 01
#   FROM_EXP=3 bash run_all.sh  # resume from a specific experiment

set -euo pipefail

DEVICE="${DEVICE:-cuda}"
FROM_EXP="${FROM_EXP:-1}"

run_if() {
    local n=$1; shift
    if [ "$n" -ge "$FROM_EXP" ]; then
        echo "[run_all] === Experiment $n ==="
        "$@"
    else
        echo "[run_all] Skipping experiment $n"
    fi
}

run_if 1 python experiments/01_norm_profiling.py --model all --device "$DEVICE"
run_if 2 python experiments/02_contrastive_extraction.py --model all --behavior all --device "$DEVICE"
run_if 3 python experiments/03_k_calibration_validation.py --model all --device "$DEVICE"
run_if 4 python experiments/04_permutation_invariance.py --model all --behavior all --device "$DEVICE"
run_if 5 python experiments/05_baking_efficacy.py --model all --behavior all --device "$DEVICE"
run_if 6 python experiments/06_weight_space_alignment.py --model all --behavior all --device "$DEVICE"

echo "[run_all] All experiments complete."
