#!/bin/bash
# ============================================================================
# Solar Panel Segmentation — Experiment Matrix
#
# Each experiment runs on 1×A100 (preempt), ~13 min/epoch.
# 2h walltime ≈ 9 epochs. 4h walltime ≈ 18 epochs.
# Uncomment experiments to submit, or submit individually.
# ============================================================================

# -----------------------------------------------------------------------
# Batch 1: Config-only — run from main branch, no code changes needed
# -----------------------------------------------------------------------

# M0 — Current baseline (Swin-B, LR 3e-5, batch 16, HFlip 0.3)
# sbatch --time=480 batch/submit_solar.sh experiment_name=baseline_swinb trainer.max_epochs=50

# M1 — Swin-T (30M params, should overfit less)
sbatch --time=480 batch/submit_solar.sh experiment_name=swin_t model=mask2former_swin_t trainer.max_epochs=50

# M3 — Swin-B, lower LR (1e-5, slower convergence but better generalization)
sbatch --time=480 batch/submit_solar.sh experiment_name=swin_b_lr1e5 model.learning_rate=1.0e-5 trainer.max_epochs=50

# M4 — No horizontal flip (test if flips hurt for aerial imagery)
sbatch --time=480 batch/submit_solar.sh experiment_name=noflip_swinb data=bdappv_noflip trainer.max_epochs=50

# -----------------------------------------------------------------------
# Batch 2: Code-change experiments — each on its own branch
# -----------------------------------------------------------------------
# Branches are already pushed.  Clone into a per-branch worktree first:
#
#   BASE="/global/cfs/cdirs/m3443/usr/pmtuan/solar-panel-seg"
#   CLONE="/pscratch/sd/p/pmtuan/solar-exp-freeze"
#   if [ ! -d "$CLONE" ]; then
#       git clone "$BASE" "$CLONE"
#   fi
#   git -C "$CLONE" fetch origin feat/freeze-backbone
#   git -C "$CLONE" checkout feat/freeze-backbone
#   REPO="$CLONE" sbatch --time=480 batch/submit_solar.sh \
#       experiment_name=freeze_swinb trainer.max_epochs=100
#
#   # EMA experiment:
#   CLONE="/pscratch/sd/p/pmtuan/solar-exp-ema"
#   if [ ! -d "$CLONE" ]; then
#       git clone "$BASE" "$CLONE"
#   fi
#   git -C "$CLONE" fetch origin feat/ema
#   git -C "$CLONE" checkout feat/ema
#   REPO="$CLONE" sbatch --time=480 "$CLONE/batch/submit_solar.sh" \
#       experiment_name=ema_swinb trainer.max_epochs=50 trainer.ema_decay=0.999
# -----------------------------------------------------------------------

# -----------------------------------------------------------------------
# Batch 3: Data experiment — BDAPPV + Bradbury combined (M7)
# -----------------------------------------------------------------------
# Prerequisite: process Bradbury data first
#   uv run python scripts/process_bradbury.py
#
# Then:
# sbatch --time=480 batch/submit_solar.sh \
#   experiment_name=combined data=bradbury data.batch_size=16 trainer.max_epochs=50
