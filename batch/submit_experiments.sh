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
# These need the branch pushed first:
#   git checkout feat/freeze-backbone && git push origin feat/freeze-backbone
#   git checkout feat/ema && git push origin feat/ema
#
# To run:
#   git clone /global/cfs/cdirs/m3443/usr/pmtuan/solar-panel-seg /tmp/solar-feat-freeze
#   cd /tmp/solar-feat-freeze && git checkout feat/freeze-backbone
#   REPO=/tmp/solar-feat-freeze sbatch --time=480 batch/submit_solar.sh \
#     experiment_name=freeze_swinb trainer.max_epochs=100
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
