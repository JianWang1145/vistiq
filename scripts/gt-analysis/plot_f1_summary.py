#!/usr/bin/env python3
"""Bar chart of per-animal F1 from f1_summary.csv.

Usage:
    python scripts/plot_f1_summary.py --summary f1_summary.csv --out f1_by_animal.png
"""
import argparse

import matplotlib.pyplot as plt
import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary", required=True)
    ap.add_argument("--out", default="f1_by_animal.png")
    args = ap.parse_args()

    df = pd.read_csv(args.summary)
    overall = df[df["animal_prefix"] == "OVERALL"]["f1"].iloc[0]
    animals = df[df["animal_prefix"] != "OVERALL"].copy()
    animals["label"] = animals["animal_prefix"].str.replace("-scrib-edu-dpn", "", regex=False)

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(animals["label"], animals["f1"], color="#4C72B0")
    ax.axhline(overall, color="gray", linestyle="--", linewidth=1, label=f"Overall F1 = {overall:.2f}")

    for bar, val in zip(bars, animals["f1"]):
        ax.text(bar.get_x() + bar.get_width() / 2, val + 0.01, f"{val:.2f}", ha="center", fontsize=9)

    ax.set_ylabel("F1 score")
    ax.set_ylim(0, 1)
    ax.set_title("MicroSAM vs. ground truth: per-animal F1")
    ax.legend()
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(args.out, dpi=150)
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()
