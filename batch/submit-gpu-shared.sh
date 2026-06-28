#!/bin/bash
# ============================================================================
# Solar Panel Segmentation — Shared GPU driver (1×A100, 40 GB)
# Usage:  sbatch batch/submit-gpu-shared.sh uv run python -m solar_seg.train ...
#
# Shared QOS: 1× billing throughout, max 4h walltime, no preemption
# For longer runs (>4h), use submit_solar.sh (preempt QOS, 48h, requeue)
# ============================================================================

#SBATCH -A m2616_g
#SBATCH -C "gpu"
#SBATCH -q shared

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=32
#SBATCH --time=4:00:00
#SBATCH --signal=SIGUSR1@240
#SBATCH --requeue
#SBATCH --gpu-bind=none
#SBATCH -o slurm_logs/solar-shared-%j-%x.out
#SBATCH -e slurm_logs/solar-shared-%j-%x.err

export SLURM_CPU_BIND="cores"
export PYTHONFAULTHANDLER=1

mkdir -p slurm_logs

# Use main venv (shared across experiments)
export VIRTUAL_ENV="/global/cfs/cdirs/m3443/usr/pmtuan/solar-panel-seg/.venv"
export PATH="${VIRTUAL_ENV}/bin:${PATH}"

echo "=============================================="
echo " Job ID:   ${SLURM_JOB_ID}"
echo " QOS:      shared (max 4h)"
echo " Command:  uv run python -m solar_seg.train $@"
echo "=============================================="

srun uv run python -m solar_seg.train "$@"
