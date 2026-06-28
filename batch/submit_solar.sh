#!/bin/bash
# ============================================================================
# Solar Panel Segmentation — Preempt GPU driver (Perlmutter, 1×A100, 40GB)
#
# Usage:
#   sbatch batch/submit_solar.sh --config-name ... [hydra overrides...]
#
# Uses the preempt QOS: 1× billing first 2h, 0.25× after.
# Auto-requeues on timeout or preemption.
#
# To run from a feature branch:
#   git checkout feat/<name> && push, then:
#   REPO="/path/to/clone-on-feat-branch" sbatch batch/submit_solar.sh ...
# ============================================================================

#SBATCH -A m2616_g
#SBATCH -C "gpu"
#SBATCH -q preempt

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=2:00:00
#SBATCH --signal=USR1@240
#SBATCH --requeue
#SBATCH --gpu-bind=none
#SBATCH -o slurm_logs/solar-%j-%x.out
#SBATCH -e slurm_logs/solar-%j-%x.err

export SLURM_CPU_BIND="cores"
export PYTHONFAULTHANDLER=1

# Default repo location — override with REPO=<path> for branch runs
REPO="${REPO:-/global/cfs/cdirs/m3443/usr/pmtuan/solar-panel-seg}"

# Use main repo's venv (shared, no uv sync needed)
export VIRTUAL_ENV="/global/cfs/cdirs/m3443/usr/pmtuan/solar-panel-seg/.venv"
export PATH="${VIRTUAL_ENV}/bin:${PATH}"

echo "=============================================="
echo " Job ID:      ${SLURM_JOB_ID}"
echo " Repo:        ${REPO}"
echo " Command:     uv run python -m solar_seg.train $@"
echo "=============================================="

cd "$REPO"
mkdir -p slurm_logs

srun uv run python -m solar_seg.train "$@"
