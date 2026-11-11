#!/bin/bash
#SBATCH --job-name=dpn-finetune
#SBATCH --partition=gpu
#SBATCH --gres=gpu:b200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=dpn-finetune-%j.out
#SBATCH --error=dpn-finetune-%j.err

set -euo pipefail

# Usage:
# sbatch finetune.sbatch <env_script> <python_script> <gt_dir> <output_root>

if [[ $# -ne 4 ]]; then
    echo "Usage: sbatch $0 <env_script> <python_script> <gt_dir> <output_root>"
    exit 1
fi

# 1. Read paths from command-line arguments
ENV_SCRIPT="$(realpath "$1")"
PY_SCRIPT="$(realpath "$2")"
GT_DIR="$(realpath "$3")"
OUTPUT_ROOT="$(realpath -m "$4")"

# 2. Validate input paths
[[ -f "$ENV_SCRIPT" ]] || { echo "Environment script not found"; exit 1; }
[[ -f "$PY_SCRIPT" ]] || { echo "Python script not found"; exit 1; }
[[ -d "$GT_DIR" ]] || { echo "GT directory not found"; exit 1; }

# 3. Activate environment
source "$ENV_SCRIPT"

PYTHON="$VISTIQ_VENV/bin/python"
[[ -x "$PYTHON" ]] || { echo "Python executable not found: $PYTHON"; exit 1; }

# 4. Create a unique output directory
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RUN_NAME="finetune_dpn_${TIMESTAMP}_job${SLURM_JOB_ID}"
OUT="$OUTPUT_ROOT/$RUN_NAME"

mkdir -p "$OUT"

# 5. Log configuration
echo "====================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "Start: $(date)"
echo "Environment: $ENV_SCRIPT"
echo "Python script: $PY_SCRIPT"
echo "GT directory: $GT_DIR"
echo "Output: $OUT"
echo "====================================="

nvidia-smi

# 6. Run Fine-tuning
"$PYTHON" "$PY_SCRIPT" \
    --finetune-dpn \
    -i "$GT_DIR" \
    -o "$OUT"

echo "Finished: $(date)"
echo "Model output: $OUT/output"