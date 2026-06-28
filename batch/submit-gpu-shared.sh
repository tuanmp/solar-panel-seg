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

# Default repo — override with REPO=/path/to/clone for branch experiments
REPO="${REPO:-/global/cfs/cdirs/m3443/usr/pmtuan/solar-panel-seg}"

# Use the main repo's venv. For branch clones, symlink .venv so uv finds it.
MAIN_REPO="/global/cfs/cdirs/m3443/usr/pmtuan/solar-panel-seg"
MAIN_VENV="${MAIN_REPO}/.venv"
if [ -d "$MAIN_VENV" ] && [ "$REPO" != "$MAIN_REPO" ]; then
    # Branch clone — symlink .venv so uv uses the shared venv
    if [ ! -e "${REPO}/.venv" ]; then
        ln -s "$MAIN_VENV" "${REPO}/.venv"
    fi
elif [ -d "$MAIN_VENV" ]; then
    export VIRTUAL_ENV="$MAIN_VENV"
    export PATH="${MAIN_VENV}/bin:${PATH}"
fi

mkdir -p slurm_logs

echo "=============================================="
echo " Job ID:   ${SLURM_JOB_ID}"
echo " QOS:      shared (max 4h)"
echo " Repo:     ${REPO}"
echo " Command:  uv run python -m solar_seg.train $@"
echo "=============================================="

cd "$REPO"
srun uv run python -m solar_seg.train "$@"
