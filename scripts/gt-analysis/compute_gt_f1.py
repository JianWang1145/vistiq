#!/usr/bin/env python3
"""Compute per-cell F1 scores between MicroSAM pipeline output and human-curated
ground truth, using the manifest produced by match_gt_to_series.py.

For each GT sample:
  - the binary *.dpn.gt.tif mask is connected-component labeled into per-cell instances
    (assumes annotated cells don't touch/overlap in the mask)
  - the matched z-slice is pulled from Labels_<channel>.<channel>.tif in the pipeline
    output, per the manifest's best_series/best_channel/best_z
  - each GT cell is matched against predicted cells by IoU >= --iou-threshold;
    TP = GT cells matched, FN = GT cells with no match, FP = predicted cells
    matching no GT cell

Writes a per-slice CSV and a per-animal + overall summary (micro-averaged:
TP/FP/FN pooled across slices before computing precision/recall/F1, so slices
with more cells aren't diluted by slices with few).

Run on HPC (inside vistiq-env-gpu):

    python scripts/compute_gt_f1.py \
        --manifest gt_series_manifest.csv \
        --gt-dir "/sfs/ceph/standard/vol191/siegristlab/.../samples_for_gt" \
        --jobruns-root "/sfs/ceph/standard/vol191/siegristlab/Microsam_Segmentation/jobruns/ivey-batch/full-run" \
        --out-slices f1_per_slice.csv \
        --out-summary f1_summary.csv
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import tifffile
from skimage.measure import label as cc_label, regionprops


def load_pred_slice(jobruns_root: Path, animal_prefix: str, series: str, channel: str, z: int) -> np.ndarray:
    path = jobruns_root / f"{animal_prefix}.lif" / animal_prefix / series / f"Labels_{channel}.{channel}.tif"
    stack = tifffile.imread(path)
    return stack[z]


def _iou(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    intersection = np.logical_and(mask_a, mask_b).sum()
    if intersection == 0:
        return 0.0
    union = np.logical_or(mask_a, mask_b).sum()
    return intersection / union


def compute_slice_counts(gt_instance: np.ndarray, pred_instance: np.ndarray, threshold: float) -> dict:
    gt_labels = set(np.unique(gt_instance)) - {0}
    pred_labels = set(np.unique(pred_instance)) - {0}

    if not gt_labels or not pred_labels:
        return {
            "n_gt_cells": len(gt_labels), "n_pred_cells": len(pred_labels),
            "tp": 0, "fp": len(pred_labels), "fn": len(gt_labels),
        }

    gt_props = {p.label: p for p in regionprops(gt_instance)}
    pred_props = {p.label: p for p in regionprops(pred_instance)}

    matched_gt = set()
    matched_pred = set()
    for gl, gp in gt_props.items():
        gy0, gx0, gy1, gx1 = gp.bbox
        for pl, pp in pred_props.items():
            py0, px0, py1, px1 = pp.bbox
            # Cheap bbox-overlap pre-check before touching full-size masks.
            if gx1 <= px0 or px1 <= gx0 or gy1 <= py0 or py1 <= gy0:
                continue
            iou = _iou(gt_instance == gl, pred_instance == pl)
            if iou >= threshold:
                matched_gt.add(gl)
                matched_pred.add(pl)

    tp = len(matched_gt)
    fn = len(gt_labels) - tp
    fp = len(pred_labels) - len(matched_pred)
    return {"n_gt_cells": len(gt_labels), "n_pred_cells": len(pred_labels), "tp": tp, "fp": fp, "fn": fn}


def prf1(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    precision = tp / (tp + fp) if (tp + fp) > 0 else float("nan")
    recall = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else float("nan")
    return precision, recall, f1


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", required=True, type=Path)
    ap.add_argument("--gt-dir", required=True, type=Path)
    ap.add_argument("--jobruns-root", required=True, type=Path)
    ap.add_argument("--iou-threshold", type=float, default=0.5)
    ap.add_argument("--out-slices", required=True, type=Path)
    ap.add_argument("--out-summary", required=True, type=Path)
    args = ap.parse_args()

    manifest = pd.read_csv(args.manifest)

    n_skip = int((manifest["status"] != "OK").sum())
    if n_skip:
        print(f"Skipping {n_skip} manifest rows not marked OK")
    manifest = manifest[manifest["status"] == "OK"].copy()

    slice_rows = []
    for _, row in manifest.iterrows():
        gt_path = args.gt_dir / row["gt_file"]
        gt_binary = tifffile.imread(gt_path) > 0
        gt_instance = cc_label(gt_binary, connectivity=2)

        pred_slice = load_pred_slice(
            args.jobruns_root, row["animal_prefix"], row["best_series"], row["best_channel"], int(row["best_z"])
        )

        counts = compute_slice_counts(gt_instance, pred_slice, args.iou_threshold)
        precision, recall, f1 = prf1(counts["tp"], counts["fp"], counts["fn"])

        slice_rows.append({
            "gt_file": row["gt_file"],
            "animal_prefix": row["animal_prefix"],
            "scene": row["scene"],
            "z": row["best_z"],
            "series": row["best_series"],
            "channel": row["best_channel"],
            **counts,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        })
        print(
            f"{row['gt_file']}: gt_cells={counts['n_gt_cells']} pred_cells={counts['n_pred_cells']} "
            f"tp={counts['tp']} fp={counts['fp']} fn={counts['fn']} f1={f1:.3f}"
        )

    slices_df = pd.DataFrame(slice_rows)
    slices_df.to_csv(args.out_slices, index=False)
    print(f"\nWrote {len(slices_df)} per-slice rows to {args.out_slices}")

    summary_rows = []
    for animal, group in slices_df.groupby("animal_prefix"):
        tp, fp, fn = group["tp"].sum(), group["fp"].sum(), group["fn"].sum()
        precision, recall, f1 = prf1(tp, fp, fn)
        summary_rows.append({
            "animal_prefix": animal, "n_slices": len(group),
            "tp": tp, "fp": fp, "fn": fn,
            "precision": precision, "recall": recall, "f1": f1,
            "mean_per_slice_f1": group["f1"].mean(),
        })

    tp, fp, fn = slices_df["tp"].sum(), slices_df["fp"].sum(), slices_df["fn"].sum()
    precision, recall, f1 = prf1(tp, fp, fn)
    summary_rows.append({
        "animal_prefix": "OVERALL", "n_slices": len(slices_df),
        "tp": tp, "fp": fp, "fn": fn,
        "precision": precision, "recall": recall, "f1": f1,
        "mean_per_slice_f1": slices_df["f1"].mean(),
    })

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(args.out_summary, index=False)
    print(f"Wrote summary to {args.out_summary}")
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()
