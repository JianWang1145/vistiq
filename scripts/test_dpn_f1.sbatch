#!/bin/bash
#SBATCH --job-name=dpn-f1-test
#SBATCH --partition=gpu
#SBATCH --gres=gpu:b200:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=dpn-test-%j.out
#SBATCH --error=dpn-test-%j.err

set -euo pipefail

# Arguments
RUN_DIR="$1"
GT_DIR="$2"
MODEL="$3"

# Activate B200 environment
source /scratch/fcc6wd/vistiq-b200/env-b200.sh

cd /scratch/fcc6wd/vistiq

echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "Training directory: $RUN_DIR"
echo "Fine-tuned model: $MODEL"
echo "Ground truth: $GT_DIR"

# Check required files
test -s "$MODEL"
test -f "$RUN_DIR/split.txt"

# Run Test Set segmentation and F1 evaluation
"$VISTIQ_VENV/bin/python" -u scripts/full_pipeline_batch.py \
    --eval-dpn-test \
    --dpn-checkpoint "$MODEL" \
    -i "$GT_DIR" \
    -o "$RUN_DIR"

echo "Test evaluation completed successfully."
