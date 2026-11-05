# UVA HPC B200 environment and GT plotting

This branch is based on `main` and includes the three GT analysis utilities
copied from `aw-batch-test`:

- `scripts/gt-analysis/match_gt_to_series.py`
- `scripts/gt-analysis/compute_gt_f1.py`
- `scripts/gt-analysis/plot_f1_summary.py`

The plot script needs `matplotlib` and `pandas`; matplotlib is now declared
in `pyproject.toml` under the `gt-analysis` optional dependency group.

## Tested environment (UVA B200)

- Python 3.12.12
- PyTorch 2.9.1+cu128; torchvision 0.24.1+cu128
- NumPy 2.1.3
- micro_sam 1.8.14
- CUDA 12.8 PyTorch wheels

`requirements-b200.in` is a *top-level pinned specification*, **not** the
complete lock. The complete `requirements-b200.lock.txt` from the original HPC
workspace is not available to this GitHub branch yet; do not assume the source
spec alone reproduces all transitive versions.

Example starting installation using `uv` from this repository:

```bash
uv venv --python 3.12 venv-b200
uv pip install --python venv-b200/bin/python \
  --index-url https://download.pytorch.org/whl/cu128 \
  'torch==2.9.1+cu128' 'torchvision==0.24.1+cu128'
uv pip install --python venv-b200/bin/python -r requirements-b200.in
uv pip check --python venv-b200/bin/python
```

On the HPC with the original resolved lockfile, prefer the **tested** lockfile,
which records the exact versions of the full environment. Check that CUDA
PyTorch wheel sources remain configured correctly when syncing.

## Generate the per-animal GT F1 bar plot

After `f1_summary.csv` has been produced:

```bash
export MPLBACKEND=Agg
venv-b200/bin/python scripts/gt-analysis/plot_f1_summary.py \
  --summary results/GT_Analysis_24h_All/f1_summary.csv \
  --out results/GT_Analysis_24h_All/f1_by_animal.png
```

The GT matching/scoring scripts come from an older pipeline format:
`match_gt_to_series.py` expects `Preprocessed_<channel>.<channel>.tif`
and `compute_gt_f1.py` expects `Labels_<channel>.<channel>.tif`.
When evaluating newer `Animal-N-...scene-S.Dpn.tif` outputs, supply the
matching manifest and legacy-compatible links as in the HPC workflow. The
plotting script accepts the resulting summary CSV directly.

The pipeline modifications made locally on the HPC to remove image enhancement
are **not** included in this dependency-only branch.
