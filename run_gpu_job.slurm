#!/bin/sh
#SBATCH --account=studiegrupper-cogito
#SBATCH --job-name=AI_MOTION_PREDICTOR
#SBATCH --time=00:30:00
#SBATCH --partition=GPUQ
#SBATCH --gres=gpu:v100:1
#SBATCH --mem=32GB
#SBATCH --nodes=1
#SBATCH --output=slurm_outputs/output_combined.txt
#SBATCH --error=slurm_outputs/output_combined.err
#SBATCH --mail-user=aleksmir@ntnu.no
#SBATCH --mail-type=ALL

set -eu

cd "$SLURM_SUBMIT_DIR"

if [ -f "$HOME/.local/bin/env" ]; then
  . "$HOME/.local/bin/env"
fi

if command -v module >/dev/null 2>&1; then
  module purge
  module load Python/3.12.3-GCCcore-13.3.0
fi

if [ -z "${RUN_CMD:-}" ]; then
  echo "No command provided."
  echo "Submit with: sbatch --export=ALL,RUN_CMD='uv run python <script>.py' run_gpu_job.slurm"
  exit 1
fi

sh -lc "$RUN_CMD"
