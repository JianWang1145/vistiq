#!/usr/bin/env python3
"""Match ground-truth 2D slices to their source (Series, channel, z) in the batch
pipeline output, by correlating each GT sample's raw image against every
Preprocessed_<channel>.<channel>.tif slice available for that animal.

Run on HPC (inside vistiq-env-gpu) where the jobruns output actually lives:

    python match_gt_to_series.py \
        --gt-dir vol191/siegristlab/Microsam_Segmentation/24h/AkhGal4 x OR Susie/Scrib488 Dpn555 EdU 647/Raw files/samples_for_gt \
        --jobruns-root /sfs/ceph/standard/vol191/siegristlab/Microsam_Segmentation/jobruns/ivey-batch/full-run \
        --out gt_series_manifest.csv
"""
import argparse
import re
from pathlib import Path

import numpy as np
import tifffile

GT_PATTERN = re.compile(r"^(?P<prefix>.+)-scene-(?P<scene>\d+)\.z-(?P<z>\d+)\.dpn\.gt\.tif$")
CHANNELS = ["Blue", "Green", "Red"]


def best_z_correlation(raw: np.ndarray, stack: np.ndarray) -> np.ndarray:
    """Vectorized Pearson correlation of `raw` (H,W) against every z-slice in `stack` (Z,H,W)."""
    raw_flat = raw.ravel().astype(np.float64)
    raw_flat -= raw_flat.mean()
    raw_norm = np.linalg.norm(raw_flat)
    if raw_norm == 0:
        return np.zeros(stack.shape[0])

    stack_flat = stack.reshape(stack.shape[0], -1).astype(np.float64)
    stack_flat -= stack_flat.mean(axis=1, keepdims=True)
    stack_norm = np.linalg.norm(stack_flat, axis=1)
    stack_norm[stack_norm == 0] = np.inf

    return (stack_flat @ raw_flat) / (stack_norm * raw_norm)


def find_series_dirs(jobruns_root: Path, animal_prefix: str) -> list[Path]:
    animal_dir = jobruns_root / f"{animal_prefix}.lif" / animal_prefix
    if not animal_dir.is_dir():
        return []
    return sorted(p for p in animal_dir.glob("Series*") if p.is_dir())


def build_animal_cache(jobruns_root: Path, animal_prefix: str) -> dict:
    """Load and normalize every Preprocessed_<channel>.<channel>.tif for this animal once.

    Returns {(series_name, channel): (mean_centered_flat_stack, per_z_norm)} so that
    per-GT-sample correlation is just a matmul against already-prepared data, with no
    repeated disk reads for the same animal.
    """
    cache = {}
    for series_dir in find_series_dirs(jobruns_root, animal_prefix):
        for channel in CHANNELS:
            pre_path = series_dir / f"Preprocessed_{channel}.{channel}.tif"
            if not pre_path.exists():
                continue
            stack = tifffile.imread(pre_path)
            stack_flat = stack.reshape(stack.shape[0], -1).astype(np.float64)
            stack_flat -= stack_flat.mean(axis=1, keepdims=True)
            stack_norm = np.linalg.norm(stack_flat, axis=1)
            stack_norm[stack_norm == 0] = np.inf
            cache[(series_dir.name, channel)] = (stack_flat, stack_norm)
    return cache


def match_gt_sample(raw_path: Path, animal_cache: dict) -> dict:
    raw = tifffile.imread(raw_path)
    raw_flat = raw.ravel().astype(np.float64)
    raw_flat -= raw_flat.mean()
    raw_norm = np.linalg.norm(raw_flat)

    candidates = []  # (series_name, channel, best_z, best_corr)
    for (series_name, channel), (stack_flat, stack_norm) in animal_cache.items():
        if raw_norm == 0:
            corr = np.zeros(stack_flat.shape[0])
        else:
            corr = (stack_flat @ raw_flat) / (stack_norm * raw_norm)
        z_best = int(np.argmax(corr))
        candidates.append((series_name, channel, z_best, float(corr[z_best])))

    candidates.sort(key=lambda c: -c[3])
    if not candidates:
        return {
            "best_series": None, "best_channel": None, "best_z": None,
            "best_corr": None, "second_corr": None, "n_candidates": 0,
        }

    best = candidates[0]
    second_corr = candidates[1][3] if len(candidates) > 1 else None
    return {
        "best_series": best[0], "best_channel": best[1], "best_z": best[2],
        "best_corr": best[3], "second_corr": second_corr, "n_candidates": len(candidates),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--gt-dir", required=True, type=Path)
    ap.add_argument("--jobruns-root", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument(
        "--min-confidence", type=float, default=0.15,
        help="Flag matches below this correlation as needing manual review.",
    )
    args = ap.parse_args()

    rows = []
    gt_files = sorted(args.gt_dir.glob("*.dpn.gt.tif"))
    print(f"Found {len(gt_files)} ground-truth files in {args.gt_dir}")

    # Group by animal so each animal's Preprocessed_*.tif stacks are read once,
    # not once per GT sample.
    by_animal: dict[str, list[re.Match]] = {}
    unmatched = []
    for gt_path in gt_files:
        m = GT_PATTERN.match(gt_path.name)
        if not m:
            unmatched.append(gt_path)
            continue
        by_animal.setdefault(m.group("prefix"), []).append((gt_path, m))
    for gt_path in unmatched:
        print(f"  SKIP (name doesn't match pattern): {gt_path.name}")

    for animal_prefix, entries in sorted(by_animal.items()):
        print(f"\n=== {animal_prefix}: caching Preprocessed stacks for {len(entries)} GT samples ===")
        animal_cache = build_animal_cache(args.jobruns_root, animal_prefix)
        if not animal_cache:
            print(f"  WARNING: no Preprocessed_*.tif found for {animal_prefix} under {args.jobruns_root}")

        for gt_path, m in entries:
            scene = int(m.group("scene"))
            z_requested = int(m.group("z"))
            raw_path = gt_path.with_name(gt_path.name.replace(".gt.tif", ".tif"))
            if not raw_path.exists():
                print(f"  SKIP (no raw counterpart): {gt_path.name}")
                continue

            result = match_gt_sample(raw_path, animal_cache)
            flag = (
                "REVIEW"
                if result["best_corr"] is None or result["best_corr"] < args.min_confidence
                else "OK"
            )
            row = {
                "gt_file": gt_path.name,
                "raw_file": raw_path.name,
                "animal_prefix": animal_prefix,
                "scene": scene,
                "z_requested": z_requested,
                "status": flag,
                **result,
            }
            rows.append(row)
            print(
                f"  {gt_path.name}: series={result['best_series']} "
                f"channel={result['best_channel']} z={result['best_z']} "
                f"corr={result['best_corr']} [{flag}]"
            )

    import csv
    with open(args.out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else [])
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nWrote {len(rows)} rows to {args.out}")
    n_review = sum(1 for r in rows if r["status"] == "REVIEW")
    if n_review:
        print(f"WARNING: {n_review} samples flagged for manual review (low confidence match)")


if __name__ == "__main__":
    main()
