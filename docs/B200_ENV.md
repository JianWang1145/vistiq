# UVA HPC B200 environment (dependency configuration only)

This `b200-uv-dependencies` branch is based on `main`. It only adds or updates
environment configuration. It deliberately does **not** copy the analysis or
plotting scripts from `aw-batch-test`: keep running those from the original
`aw-batch-test` worktree, and keep the original `main` pipeline intact.

## Tested environment (UVA B200)

- Python 3.12.12
- PyTorch 2.9.1+cu128 and torchvision 0.24.1+cu128
- NumPy 2.1.3
- micro_sam 1.8.14
- CUDA 12.8 PyTorch wheels
- `matplotlib` and `pandas` for `aw-batch-test` GT analysis plots

The repository's `pyproject.toml` has an optional `gt-analysis` dependency
group for plotting, and `requirements-b200.in` pins the primary packages.

**Important:** `requirements-b200.in` is only a top-level requirements
specification. The fully resolved `requirements-b200.lock.txt` generated on
the UVA HPC is not yet included in this branch. It is required to preserve the
exact versions of all transitive dependencies.

## Starting installation with uv

```bash
uv venv --python 3.12 venv-b200
uv pip install --python venv-b200/bin/python \
  --index-url https://download.pytorch.org/whl/cu128 \
  'torch==2.9.1+cu128' 'torchvision==0.24.1+cu128'
uv pip install --python venv-b200/bin/python -r requirements-b200.in
uv pip check --python venv-b200/bin/python
```

If possible, use the original fully resolved HPC lock rather than resolving
transitive packages anew.

## Running GT analysis and F1 plots

Use the existing `aw-batch-test` worktree on HPC; these files are **not**
duplicated in this branch:

```bash
cd /scratch/fcc6wd/vistiq-b200
source env-b200.sh
AW="$PWD/vistiq-aw/scripts/gt-analysis"
OUT="$PWD/results/GT_Analysis_24h_All"
GT="/project/vistiqlab/brain/24h/AkhGal4 x OR Susie/Scrib488 Dpn555 EdU 647/Raw files/samples_for_gt"

"$VISTIQ_VENV/bin/python" "$AW/compute_gt_f1.py" \
  --manifest "$OUT/gt_series_manifest.csv" \
  --gt-dir "$GT" --jobruns-root "$OUT/legacy" \
  --iou-threshold 0.5 \
  --out-slices "$OUT/f1_per_slice.csv" \
  --out-summary "$OUT/f1_summary.csv"

MPLBACKEND=Agg "$VISTIQ_VENV/bin/python" "$AW/plot_f1_summary.py" \
  --summary "$OUT/f1_summary.csv" \
  --out "$OUT/f1_by_animal.png"
```

The original GT scoring script expects the older pipeline label filenames.
Generate the matching manifest and legacy-compatible symlinks first for the
new `Animal-N-...scene-S.Dpn.tif` outputs.

The local HPC code changes to remove image enhancement during tissue
preprocessing are not included in this dependency-only branch.
