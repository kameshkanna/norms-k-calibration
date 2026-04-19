#!/usr/bin/env bash
# Run all experiments in order across all models and behaviours.
# Usage: bash run_all.sh [--device cuda] [--skip-01] [--from 03]
#
# Experiments 03-06 accept --model and --behavior and are looped here.
# Experiment 01 (norm profiling) and 02 (contrastive extraction) run with --model all --behavior all.

set -euo pipefail

DEVICE="${DEVICE:-cuda}"
MODELS="llama qwen mistral gemma"
BEHAVIORS="refusal_calibration formality verbosity_control uncertainty_expression sycophancy_suppression"
FROM_EXP="${FROM_EXP:-1}"

log() { echo "[run_all] $*"; }

run_if() {
    local exp_num=$1; shift
    if [ "$exp_num" -ge "$FROM_EXP" ]; then
        log "=== Experiment $exp_num ==="
        "$@"
    else
        log "Skipping experiment $exp_num (FROM_EXP=$FROM_EXP)"
    fi
}

run_if 1 python experiments/01_norm_profiling.py \
    --model all --device "$DEVICE"

run_if 2 python experiments/02_contrastive_extraction.py \
    --model all --behavior all --device "$DEVICE"

if [ "$FROM_EXP" -le 3 ]; then
    log "=== Experiment 3 ==="
    for model in $MODELS; do
        log "  k_calibration: $model"
        python experiments/03_k_calibration_validation.py \
            --model "$model" --device "$DEVICE"
    done
fi

if [ "$FROM_EXP" -le 4 ]; then
    log "=== Experiment 4 ==="
    for model in $MODELS; do
        for behavior in $BEHAVIORS; do
            log "  permutation: $model / $behavior"
            python experiments/04_permutation_invariance.py \
                --model "$model" --behavior "$behavior" --device "$DEVICE"
        done
    done
fi

if [ "$FROM_EXP" -le 5 ]; then
    log "=== Experiment 5 ==="
    for model in $MODELS; do
        for behavior in $BEHAVIORS; do
            log "  efficacy: $model / $behavior"
            python experiments/05_baking_efficacy.py \
                --model "$model" --behavior "$behavior" --device "$DEVICE"
        done
    done
fi

if [ "$FROM_EXP" -le 6 ]; then
    log "=== Experiment 6 ==="
    for model in $MODELS; do
        for behavior in $BEHAVIORS; do
            log "  weight_alignment: $model / $behavior"
            python experiments/06_weight_space_alignment.py \
                --model "$model" --behavior "$behavior" --device "$DEVICE"
        done
    done
fi

log "All experiments complete."
