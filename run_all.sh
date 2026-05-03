#!/usr/bin/env bash
# run_all.sh — Master runner for the full K-calibration experiment suite.
#
# Usage:
#   bash run_all.sh                    # full run, all models
#   bash run_all.sh --paper-models     # paper model set (4 mid-size models)
#   bash run_all.sh --dev              # fast dev set (5 small models)
#   bash run_all.sh --tier mid         # one size tier: small | mid | large
#   bash run_all.sh --model llama_8b   # single model
#   bash run_all.sh --overwrite        # ignore checkpoints, recompute all
#   bash run_all.sh --skip-generation  # skip E6 (requires OPENAI_API_KEY)
#   bash run_all.sh --dry-run          # print commands without executing
#   bash run_all.sh --from-stage 4     # resume from a specific stage number
#
# Script → stage mapping:
#   Stage 1  : 01_norm_profiling.py
#   Stage 2  : 02_contrastive_extraction.py
#   Stage 3  : 03_k_calibration_validation.py
#   Stage 4  : E4_weight_alignment_v2.py
#   Stage 5  : E5_permutation_invariance_v2.py
#   Stage 6  : E7_k_sensitivity_curve.py
#   Stage 7  : E6_generation_quality.py         (requires OPENAI_API_KEY)
#   Stage 8  : 08_directional_fidelity_analysis.py
#   Stage 9  : 09_partial_correlation_analysis.py
#   Stage 10 : aggregate + figures pass
#
# Note: stages 8–9 are post-processing; they read existing CSVs and do not
# need GPU. Stage 10 runs --aggregate-only / --plot-only passes for all scripts.

set -euo pipefail

# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

MODEL_ARG="all"
OVERWRITE=""
SKIP_GENERATION=false
DRY_RUN=false
FROM_STAGE=1
DEVICE="cuda"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --paper-models)     MODEL_ARG="wss";   shift ;;
        --dev)              MODEL_ARG="dev";   shift ;;
        --tier)             MODEL_ARG="tier:$2"; shift 2 ;;
        --model)            MODEL_ARG="$2";    shift 2 ;;
        --overwrite)        OVERWRITE="--overwrite"; shift ;;
        --skip-generation)  echo "Note: --skip-generation is a no-op; generation eval lives in AVAW."; shift ;;
        --dry-run)          DRY_RUN=true; shift ;;
        --from-stage)       FROM_STAGE="$2"; shift 2 ;;
        --device)           DEVICE="$2"; shift 2 ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXP="$SCRIPT_DIR/experiments"
PYTHON="${PYTHON:-python3}"

# ---------------------------------------------------------------------------
# Resolve MODEL_ARG into a comma-separated key list (or "all").
# All experiment scripts accept: --model all | --model key1,key2,...
# Groups and tiers are expanded here so no script needs to know about them.
# ---------------------------------------------------------------------------
case "$MODEL_ARG" in
    all)
        MODEL_FLAG="--model all"
        ;;
    wss)
        MODEL_FLAG="--model llama_8b,qwen_7b,mistral_7b,gemma_9b"
        ;;
    dev)
        MODEL_FLAG="--model llama_3b,qwen_3b,mistral_7b,gemma_2b,phi_mini"
        ;;
    tier:small)
        MODEL_FLAG="--model llama_1b,llama_3b,qwen_3b,gemma_2b,phi_mini"
        ;;
    tier:mid)
        MODEL_FLAG="--model llama_8b,qwen_7b,mistral_7b,gemma_9b"
        ;;
    tier:large)
        MODEL_FLAG="--model llama_70b,qwen_14b,qwen_32b,qwen_72b,mixtral_8x7b,gemma_27b,phi_medium"
        ;;
    *)
        # Single key or already comma-separated list passed via --model
        MODEL_FLAG="--model $MODEL_ARG"
        ;;
esac

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

log()  { echo "[$(date '+%H:%M:%S')] $*"; }
stage_log() { log ""; log "=== Stage $1: $2 ==="; }

run() {
    log "  $ $*"
    if [[ "$DRY_RUN" == "true" ]]; then
        echo "    [DRY RUN — skipped]"
        return 0
    fi
    "$@"
}

should_run() {
    local stage_num="$1"
    [[ "$stage_num" -ge "$FROM_STAGE" ]]
}

# ---------------------------------------------------------------------------
# Pre-flight
# ---------------------------------------------------------------------------

log "=== K-Calibration Experiment Suite ==="
log "Models : $MODEL_ARG"
log "Device : $DEVICE"
log "Overwrite: ${OVERWRITE:-no}"
log "Dry-run  : $DRY_RUN"
log "From stage: $FROM_STAGE"

if ! $PYTHON -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
    log "WARNING: CUDA unavailable — large models will be very slow on CPU."
fi

# ---------------------------------------------------------------------------
# Stage 1 — Norm profiling
#   Computes per-layer μ̄_l and K_l = μ̄_l/√d.
#   Produces: results/k_calibration/{model}_k_vs_spectral.csv
# ---------------------------------------------------------------------------
if should_run 1; then
    stage_log 1 "Norm profiling (01_norm_profiling.py)"
    run $PYTHON "$EXP/01_norm_profiling.py" \
        $MODEL_FLAG $OVERWRITE --device "$DEVICE"
fi

# ---------------------------------------------------------------------------
# Stage 2 — Contrastive PCA direction extraction
#   Produces: results/pca_directions/{model}/{behavior}/variance_explained.csv
# ---------------------------------------------------------------------------
if should_run 2; then
    stage_log 2 "Contrastive PCA extraction (02_contrastive_extraction.py)"
    run $PYTHON "$EXP/02_contrastive_extraction.py" \
        $MODEL_FLAG $OVERWRITE --device "$DEVICE"
fi

# ---------------------------------------------------------------------------
# Stage 3 — K-calibration validation (spectral correlation)
#   Produces: results/k_calibration/{model}_correlation.json
# ---------------------------------------------------------------------------
if should_run 3; then
    stage_log 3 "K-calibration validation (03_k_calibration_validation.py)"
    run $PYTHON "$EXP/03_k_calibration_validation.py" \
        $MODEL_FLAG $OVERWRITE --device "$DEVICE"
fi

# ---------------------------------------------------------------------------
# Stage 4 — Weight alignment v2  (bootstrap CIs + random structured control)
#   Produces: results/weight_alignment_v2/{model}/{behavior}/alignment_bootstrap.csv
# ---------------------------------------------------------------------------
if should_run 4; then
    stage_log 4 "Weight alignment v2 (E4_weight_alignment_v2.py)"
    run $PYTHON "$EXP/E4_weight_alignment_v2.py" \
        $MODEL_FLAG $OVERWRITE --device "$DEVICE"
fi

# ---------------------------------------------------------------------------
# Stage 5 — Permutation invariance v2  (20 seeds + orbit-averaged probe)
#   Produces: results/permutation_invariance_v2/{model}/{behavior}/
# ---------------------------------------------------------------------------
if should_run 5; then
    stage_log 5 "Permutation invariance v2 (E5_permutation_invariance_v2.py)"
    run $PYTHON "$EXP/E5_permutation_invariance_v2.py" \
        $MODEL_FLAG $OVERWRITE --device "$DEVICE"
fi

# ---------------------------------------------------------------------------
# Stage 6 — K sensitivity curve  (κ sweep 0.01→50, 15 log-spaced points)
#   Produces: results/k_sensitivity/{model}/sensitivity_curve.csv
# ---------------------------------------------------------------------------
if should_run 6; then
    stage_log 6 "K sensitivity curve (E7_k_sensitivity_curve.py)"
    run $PYTHON "$EXP/E7_k_sensitivity_curve.py" \
        $MODEL_FLAG $OVERWRITE --device "$DEVICE"
fi

# Stage 7 (generation quality) removed — downstream eval lives in the AVAW repo.
# Pull generation results from AVAW and place in results/generation_quality/ to
# include them in paper figures without re-running here.

# ---------------------------------------------------------------------------
# Stage 8 — Directional fidelity aggregate  (post-processing, no GPU)
#   Reads: results/efficacy/*/comparison.csv
#   Produces: results/directional_fidelity/aggregate_table.csv + figures
# ---------------------------------------------------------------------------
if should_run 8; then
    stage_log 8 "Directional fidelity aggregate (08_directional_fidelity_analysis.py)"
    run $PYTHON "$EXP/08_directional_fidelity_analysis.py"
fi

# ---------------------------------------------------------------------------
# Stage 9 — Partial correlation analysis  (F2 depth-confound control, no GPU)
#   Reads: results/k_calibration/*_k_vs_spectral.csv
#   Produces: results/partial_correlation/latex_partial_table.tex
# ---------------------------------------------------------------------------
if should_run 9; then
    stage_log 9 "Partial correlation analysis (09_partial_correlation_analysis.py)"
    run $PYTHON "$EXP/09_partial_correlation_analysis.py"
fi

# ---------------------------------------------------------------------------
# Stage 10 — Aggregate passes  (--aggregate-only / --plot-only, no GPU)
# ---------------------------------------------------------------------------
if should_run 10; then
    stage_log 10 "Aggregate summaries and figures"
    run $PYTHON "$EXP/E4_weight_alignment_v2.py"        --aggregate-only
    run $PYTHON "$EXP/E5_permutation_invariance_v2.py"  --aggregate-only
    run $PYTHON "$EXP/E7_k_sensitivity_curve.py"        --plot-only
fi

# ---------------------------------------------------------------------------
# Done
# ---------------------------------------------------------------------------

log ""
log "=== All stages complete ==="
log ""
log "Key outputs:"
log "  results/k_calibration/          — per-layer K values, spectral correlations"
log "  results/weight_alignment_v2/    — alignment ratios, bootstrap CIs"
log "  results/permutation_invariance_v2/ — permutation resilience, orbit probe"
log "  results/k_sensitivity/          — κ sensitivity curves"
log "  results/partial_correlation/    — depth-controlled spectral correlations"
log "  figures/                        — all PDFs ready for LaTeX inclusion"
