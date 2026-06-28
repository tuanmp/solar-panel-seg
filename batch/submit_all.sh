#!/bin/bash
# ============================================================================
# Submit solar panel segmentation experiments to Perlmutter
# ============================================================================

set -euo pipefail

PROJECT_ROOT="/global/cfs/cdirs/m3443/usr/pmtuan/solar-panel-seg"
VENV="${PROJECT_ROOT}/.venv"
export VIRTUAL_ENV="$VENV"
export PATH="${VENV}/bin:${PATH}"

# Shared QOS:   max 4h  →  ~18 epochs @ 13 min/epoch
# Preempt QOS:  max 48h →  ~220 epochs, 0.25× billing after 2h
# Add --time=N (minutes) to override

# ─── CONFIG-ONLY EXPERIMENTS (run from main) ──────────────────────────────

# M1 — Swin-T (30M params, should generalize better)
sbatch --time=240 -q shared batch/submit-gpu-shared.sh \
    experiment_name=swin_t \
    model=mask2former_swin_t \
    trainer.max_epochs=18

# M3 — Swin-B with lower LR (1e-5)
sbatch --time=240 -q shared batch/submit-gpu-shared.sh \
    experiment_name=swin_b_lr1e5 \
    model.learning_rate=1.0e-5 \
    trainer.max_epochs=18

# M4 — No horizontal flip
sbatch --time=240 -q shared batch/submit-gpu-shared.sh \
    experiment_name=noflip_swinb \
    data=bdappv_noflip \
    trainer.max_epochs=18

# ─── FULL 50-EPOCH RUNS (use preempt QOS) ─────────────────────────────────
# Uncomment for the best config after screening:

# M1-50 — Swin-T full run
# sbatch --time=720 -q preempt batch/submit_solar.sh \
#     experiment_name=swin_t_50ep \
#     model=mask2former_swin_t \
#     trainer.max_epochs=50

# ─── BRANCH EXPERIMENTS (clone once, then submit) ──────────────────────────
#
# Freeze backbone:
#   CLONE="/pscratch/sd/p/pmtuan/solar-exp-freeze"
#   if [ ! -d "$CLONE" ]; then
#       git clone "$PROJECT_ROOT" "$CLONE"
#   fi
#   git -C "$CLONE" fetch origin feat/freeze-backbone
#   git -C "$CLONE" checkout feat/freeze-backbone
#   cd "$CLONE"
#   sbatch --time=240 -q shared batch/submit-gpu-shared.sh \
#       experiment_name=freeze_swinb \
#       model=mask2former_swin_b_frozen \
#       trainer.max_epochs=100
#
# EMA:
#   CLONE="/pscratch/sd/p/pmtuan/solar-exp-ema"
#   if [ ! -d "$CLONE" ]; then
#       git clone "$PROJECT_ROOT" "$CLONE"
#   fi
#   git -C "$CLONE" fetch origin feat/ema
#   git -C "$CLONE" checkout feat/ema
#   cd "$CLONE"
#   sbatch --time=240 -q shared batch/submit-gpu-shared.sh \
#       experiment_name=ema_swinb \
#       trainer.ema_decay=0.999 \
#       trainer.max_epochs=50

echo ""
echo "Submitted. Monitor with:  squeue -u \$USER"
echo "MLflow:                   mlflow ui --backend-store-uri mlruns"
