#!/bin/bash
# ============================================================================
# Solar Panel Segmentation — Preempt GPU driver (Perlmutter, 1×A100)
# Usage:  sbatch batch/submit_solar.sh [hydra overrides...]
# ============================================================================

#SBATCH -A m2616_g
#SBATCH -C "gpu"
#SBATCH -q preempt
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --signal=USR1@300
#SBATCH --requeue
#SBATCH --gpu-bind=none
#SBATCH -o slurm_logs/solar-%j-%x.out
#SBATCH -e slurm_logs/solar-%j-%x.err

export SLURM_CPU_BIND="cores"
export PYTHONFAULTHANDLER=1

REPO="${REPO:-/global/cfs/cdirs/m3443/usr/pmtuan/solar-panel-seg}"
MAIN_REPO="/global/cfs/cdirs/m3443/usr/pmtuan/solar-panel-seg"
MAIN_DATA="${MAIN_REPO}/data"

cd "$REPO"
mkdir -p slurm_logs

if [ "$REPO" != "$MAIN_REPO" ] && [ ! -e "$REPO/data" ]; then
    ln -s "$MAIN_DATA" "$REPO/data"
fi

uv sync 2>&1

echo "=============================================="
echo " Job ID:      ${SLURM_JOB_ID}"
echo " Repo:        ${REPO}"
echo " Command:     uv run python -m solar_seg.train $@"
echo "=============================================="

srun uv run python -m solar_seg.train "$@"
