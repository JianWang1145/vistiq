#!/usr/bin/env python3
"""Single-file full pipeline for SLURM job arrays (notebook: hierarchical-analysis-gwas.ipynb).

Same workflow as :mod:`full_pipeline` (tissue + cells + hierarchical analysis) but
accepts one input file and an output directory, for use with
``scripts/batch_process.sbatch``::

    sbatch scripts/batch_process.sbatch scripts/full_pipeline_batch.py filelist.txt /path/to/output

Or directly::

    python scripts/full_pipeline_batch.py -i "$input_file" -o "$output_dir"
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

from dpn_transforms import to_instances, to_uint8

import vistiq
from vistiq.utils import check_device

from full_pipeline import (
    DEFAULT_EMBEDDING_PATH,
    parse_channel_list,
    run_full_pipeline,
)

logger = logging.getLogger(__name__)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run tissue + cell segmentation and hierarchical analysis on one "
            "microscopy volume (.lif, .tif, etc.)."
        ),
    )
    parser.add_argument(
        "-i",
        "--input",
        required=True,
        type=Path,
        help="Input image path (.lif, .tif, etc.)",
    )
    parser.add_argument(
        "-o",
        "--output",
        required=True,
        type=Path,
        help="Output directory for label TIFFs and measurement CSVs",
    )
    parser.add_argument(
        "--scene-index",
        type=int,
        default=0,
        help="Scene index for multi-scene containers (default: 0)",
    )
    parser.add_argument(
        "--embedding-path",
        type=Path,
        default=Path(DEFAULT_EMBEDDING_PATH),
        help="Directory with MicroSAM embeddings "
        f"(default: VISTIQ_EMBEDDING_PATH or {DEFAULT_EMBEDDING_PATH!r})",
    )
    parser.add_argument(
        "--channels",
        type=str,
        default=None,
        help="Comma-separated channel names applied after loading (e.g. Dpn,InR,EdU)",
    )
    parser.add_argument(
        "--channels-from-filename",
        action="store_true",
        help="Parse channel names from the input basename: split on '-', take the "
        "last N segments (N = number of channels). Ignored when --channels is set.",
    )
    parser.add_argument(
        "--preprocess-workers",
        type=int,
        default=-1,
        help="Worker count for PreprocessFlow (default: -1, all cores)",
    )
    parser.add_argument(
        "--segment-workers",
        type=int,
        default=2,
        help="Worker count for TiledSegmentationFlow (default: 2)",
    )
    parser.add_argument(
        "--tissue-labels",
        choices=("2d", "3d"),
        default="2d",
        help=(
            "Tissue segmentation mode: 2d Z-projects before segmentation and expands "
            "lobe labels to 3D; 3d segments the full volume (default: 2d)"
        ),
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level (default: INFO)",
    )
    parser.add_argument(
    "--finetune-dpn",
    action="store_true",
    help="Fine-tune MicroSAM on Dpn cell ground truth",
    )
    parser.add_argument(
        "--dpn-checkpoint",
        type=Path,
        default=None,
        help="Fine-tuned Dpn MicroSAM model (.pth). "
             "If omitted, use pretrained MicroSAM.",
    )
    parser.add_argument(
        "--eval-dpn-test",
        action="store_true",
        help="Run Dpn test-set segmentation and F1 evaluation",
    )
    return parser.parse_args(argv)






#below is the fine tuning function(Chat wrote this and I doubled checked it to make sure it's running fine. It should be, I guess...)






def finetune_dpn(gt_dir: Path, output_dir: Path) -> None:
    """Fine-tune MicroSAM for Dpn cell instance segmentation."""
    import random
    import tifffile
    import torch
    import micro_sam.training as sam_training
    from micro_sam.util import export_custom_sam_model
    # 1. Find paired Dpn images and ground truth.
    gt_files = sorted(gt_dir.glob("*.dpn.gt.tif"))

    if len(gt_files) != 120:
        raise ValueError(
            f"Expected 120 GT images, found {len(gt_files)}"
        )

    groups = {}

    for gt in gt_files:
        raw = gt.with_name(gt.name.replace(".gt.tif", ".tif"))

        if not raw.is_file():
            raise FileNotFoundError(raw)

        if "-scene-" not in gt.name:
            raise ValueError(f"Unexpected filename: {gt.name}")

        animal = gt.name.rsplit("-scene-", 1)[0]

        groups.setdefault(animal, []).append((raw, gt))

    if len(groups) < 3:
        raise ValueError("Need at least three Animals for splitting")

    # 2. Animal-level 80% train / 10% val / 10% test split.
    animals = sorted(groups)
    random.Random(114514).shuffle(animals)

    total = len(gt_files)
    target_val = total * 0.10
    target_test = total * 0.10

    # Dynamic programming: find two disjoint Animal groups
    # whose image counts are closest to 10% each.
    # State: (number_of_val_images, number_of_test_images)
    states = {(0, 0): ((), ())}

    for animal in animals:
        count = len(groups[animal])
        new_states = dict(states)  # Assign animal to training.

        for (nv, nt), (va, ta) in states.items():
            # Assign animal to validation.
            new_states.setdefault(
                (nv + count, nt),
                (va + (animal,), ta),
            )

            # Assign animal to test.
            new_states.setdefault(
                (nv, nt + count),
                (va, ta + (animal,)),
            )

        states = new_states

    candidates = [
        (counts, selected)
        for counts, selected in states.items()
        if counts[0] > 0
        and counts[1] > 0
        and sum(counts) < total
    ]

    if not candidates:
        raise ValueError("Cannot create three nonempty Animal-level splits")

    (val_count, test_count), (selected_val, selected_test) = min(
        candidates,
        key=lambda item: (
            (item[0][0] - target_val) ** 2
            + (item[0][1] - target_test) ** 2,
            abs(item[0][0] - item[0][1]),
        ),
    )

    val_animals = set(selected_val)
    test_animals = set(selected_test)
    train_animals = set(animals) - val_animals - test_animals

    train_pairs = [
        pair
        for animal in animals if animal in train_animals
        for pair in groups[animal]
    ]

    val_pairs = [
        pair
        for animal in animals if animal in val_animals
        for pair in groups[animal]
    ]

    test_pairs = [
        pair
        for animal in animals if animal in test_animals
        for pair in groups[animal]
    ]

    print(f"Training images: {len(train_pairs)} ({len(train_pairs)/total:.1%})")
    print(f"Validation images: {len(val_pairs)} ({len(val_pairs)/total:.1%})")
    print(f"Test images: {len(test_pairs)} ({len(test_pairs)/total:.1%})")
    print("Training Animals:", sorted(train_animals))
    print("Validation Animals:", sorted(val_animals))
    print("Test Animals:", sorted(test_animals))

    # Save all three splits for reproducibility.
    output_dir.mkdir(parents=True, exist_ok=True)

    with (output_dir / "split.txt").open("w") as f:
        for split, pairs in (
            ("train", train_pairs),
            ("val", val_pairs),
            ("test", test_pairs),
        ):
            for raw, gt in pairs:
                f.write(f"{split}\t{raw.name}\t{gt.name}\n")

    # 3. Convert binary GT into instance labels. (Andy's note: This converts binary masks into labels. However, ajacent cells will be labeled as one cell. Remind yourself to validate this in the future. If you read this and Andy still haven't done this, please yell at him.)
    # 5. Build MicroSAM data loaders.
    def make_loader(pairs, is_train):
        return sam_training.default_sam_loader(
            raw_paths=[str(raw) for raw, gt in pairs],
            raw_key=None,
            label_paths=[str(gt) for raw, gt in pairs],
            label_key=None,
            is_seg_dataset=False,
            patch_shape=(512, 512),
            batch_size=1,
            pre_label_transform=to_instances,
            raw_transform=to_uint8,
            min_size=8,
            with_segmentation_decoder=True,
            is_train=is_train,
            num_workers=2,
            shuffle=is_train,
        )

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is not available")

    train_loader = make_loader(train_pairs, True)
    val_loader = make_loader(val_pairs, False)

    # 6. Fine-tune the pretrained MicroSAM. (Andy's note: this is the actual fine tuning part.)
    model_output = output_dir / "output"
    model_output.mkdir(parents=True, exist_ok=True)

    sam_training.train_sam(
        name="dpn_v1",
        model_type="vit_l_lm",
        train_loader=train_loader,
        val_loader=val_loader,
        n_epochs=20,
        early_stopping=5,
        n_objects_per_batch=15,
        n_sub_iteration=4,
        lr=1e-5,
        with_segmentation_decoder=True,
        device="cuda",
        save_root=str(model_output),
        overwrite_training=False,
    )

    # 7. Export the best checkpoint.
    best = model_output / "checkpoints" / "dpn_v1" / "best.pt"

    if not best.is_file():
        raise FileNotFoundError(best)

    exported = model_output / "dpn_v1.pth"

    export_custom_sam_model(
        checkpoint_path=str(best),
        model_type="vit_l",
        save_path=str(exported),
        with_segmentation_decoder=True,
    )

    print("Exported model:", exported)

def evaluate_dpn_test(
    gt_dir: Path,
    run_dir: Path,
    checkpoint: Path,
) -> None:
    """Evaluate pretrained and fine-tuned MicroSAM on the held-out Dpn test set."""

    import csv
    import numpy as np
    import tifffile
    import torch

    from scipy.optimize import linear_sum_assignment
    from dpn_transforms import to_uint8, to_instances
    from micro_sam.automatic_segmentation import (
        get_predictor_and_segmenter,
        automatic_instance_segmentation,
    )

    split_file = run_dir / "split.txt"
    output_dir = run_dir / "test_evaluation"

    if not split_file.is_file():
        raise FileNotFoundError(split_file)

    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required")

    # 1. Read the held-out Test images.
    test_pairs = []
    animal_sets = {"train": set(), "val": set(), "test": set()}

    for line in split_file.read_text().splitlines():
        if not line.strip():
            continue

        split, raw_name, gt_name = line.split("\t")

        if split not in animal_sets:
            raise ValueError(f"Unknown split: {split}")

        if "-scene-" not in gt_name:
            raise ValueError(f"Invalid GT filename: {gt_name}")

        animal = gt_name.rsplit("-scene-", 1)[0]
        animal_sets[split].add(animal)

        if split == "test":
            raw_path = gt_dir / raw_name
            gt_path = gt_dir / gt_name

            if not raw_path.is_file():
                raise FileNotFoundError(raw_path)

            if not gt_path.is_file():
                raise FileNotFoundError(gt_path)

            test_pairs.append((animal, raw_path, gt_path))

    if not test_pairs:
        raise ValueError("No test images found in split.txt")

    if (
        animal_sets["test"] & animal_sets["train"]
        or animal_sets["test"] & animal_sets["val"]
    ):
        raise ValueError("Animal leakage detected!")

    print("Test images:", len(test_pairs), flush=True)
    print("Test Animals:", sorted(animal_sets["test"]), flush=True)

    output_dir.mkdir(parents=True, exist_ok=True)

    # 2. Calculate instance-level TP, FP, FN.
    # A prediction counts as correct if IoU >= 0.5.
    def count_matches(gt, pred):
        gt = np.asarray(gt)
        pred = np.asarray(pred)

        if gt.shape != pred.shape or gt.ndim != 2:
            raise ValueError("GT and prediction shapes do not match")

        gt_ids, gi = np.unique(gt.ravel(), return_inverse=True)
        pred_ids, pi = np.unique(pred.ravel(), return_inverse=True)

        overlaps = np.bincount(
            gi * len(pred_ids) + pi,
            minlength=len(gt_ids) * len(pred_ids),
        ).reshape(len(gt_ids), len(pred_ids))

        g = np.flatnonzero(gt_ids > 0)
        p = np.flatnonzero(pred_ids > 0)

        n_gt = len(g)
        n_pred = len(p)

        if n_gt == 0 or n_pred == 0:
            return 0, n_pred, n_gt

        intersection = overlaps[np.ix_(g, p)].astype(float)

        gt_area = overlaps.sum(axis=1)[g]
        pred_area = overlaps.sum(axis=0)[p]

        union = (
            gt_area[:, None]
            + pred_area[None, :]
            - intersection
        )

        iou = np.divide(
            intersection,
            union,
            out=np.zeros_like(intersection),
            where=union > 0,
        )

        valid = iou >= 0.5

        # One-to-one matching, maximizing the number of valid matches.
        rows, cols = linear_sum_assignment(
            -valid.astype(np.int8)
        )

        tp = int(valid[rows, cols].sum())
        fp = n_pred - tp
        fn = n_gt - tp

        return tp, fp, fn

    def calculate_f1(tp, fp, fn):
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0

        return precision, recall, f1

    # 3. Compare pretrained and fine-tuned models.
    models = [
        ("pretrained", "vit_l_lm", None),
        ("finetuned", "vit_l", str(checkpoint)),
    ]

    results = {}

    with (output_dir / "per_image.csv").open(
        "w", newline=""
    ) as f:

        writer = csv.writer(f)

        writer.writerow([
            "model", "animal", "image",
            "tp", "fp", "fn",
            "precision", "recall", "f1",
        ])

        for model_name, model_type, model_checkpoint in models:

            print(f"Loading {model_name}", flush=True)

            predictor, segmenter = get_predictor_and_segmenter(
                model_type=model_type,
                checkpoint=model_checkpoint,
                segmentation_mode="ais",
                device="cuda",
            )

            pred_dir = output_dir / "predictions" / model_name
            pred_dir.mkdir(parents=True, exist_ok=True)

            total_tp = total_fp = total_fn = 0

            for index, (animal, raw_path, gt_path) in enumerate(
                test_pairs, 1
            ):
                raw = tifffile.imread(raw_path)
                binary_gt = tifffile.imread(gt_path)

                if raw.ndim != 2 or binary_gt.ndim != 2:
                    raise ValueError(
                        f"Expected 2D images: {raw_path.name}"
                    )

                image = to_uint8(raw)
                gt = to_instances(binary_gt)

                with torch.inference_mode():
                    pred = automatic_instance_segmentation(
                        predictor=predictor,
                        segmenter=segmenter,
                        input_path=image,
                        ndim=2,
                        verbose=False,
                    )

                pred = np.asarray(pred)

                tp, fp, fn = count_matches(gt, pred)
                precision, recall, f1 = calculate_f1(tp, fp, fn)

                total_tp += tp
                total_fp += fp
                total_fn += fn

                tifffile.imwrite(
                    pred_dir / f"{raw_path.stem}_instances.tif",
                    pred.astype(np.uint32),
                )

                writer.writerow([
                    model_name,
                    animal,
                    raw_path.name,
                    tp, fp, fn,
                    precision, recall, f1,
                ])

                f.flush()

                print(
                    f"{model_name} [{index}/{len(test_pairs)}] "
                    f"F1={f1:.4f} {raw_path.name}",
                    flush=True,
                )

            precision, recall, f1 = calculate_f1(
                total_tp, total_fp, total_fn
            )

            results[model_name] = [
                total_tp, total_fp, total_fn,
                precision, recall, f1,
            ]

            del predictor, segmenter
            torch.cuda.empty_cache()

    # 4. Save final results.
    summary_path = output_dir / "summary.csv"

    with summary_path.open("w", newline="") as f:
        writer = csv.writer(f)

        writer.writerow([
            "model", "tp", "fp", "fn",
            "precision", "recall", "f1",
        ])

        for model_name, values in results.items():
            writer.writerow([model_name, *values])

    print("\n=== FINAL TEST RESULTS ===", flush=True)

    for name, values in results.items():
        print(
            f"{name}: "
            f"Precision={values[3]:.4f}, "
            f"Recall={values[4]:.4f}, "
            f"F1={values[5]:.4f}",
            flush=True,
        )

    print("Saved:", summary_path, flush=True)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.eval_dpn_test:
        if args.dpn_checkpoint is None:
            raise ValueError(
                "--eval-dpn-test requires --dpn-checkpoint"
            )

        evaluate_dpn_test(
            args.input.expanduser().resolve(),
            args.output.expanduser().resolve(),
            args.dpn_checkpoint.expanduser().resolve(),
        )
        return 0
    if args.finetune_dpn:
        finetune_dpn(
            args.input.expanduser().resolve(),
            args.output.expanduser().resolve(),
        )
        return 0
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    )
    logging.getLogger(vistiq.__name__).setLevel(getattr(logging, args.log_level))
    logger.info("Available Torch accelerators: %s", check_device())

    input_path = args.input.expanduser().resolve()
    output_dir = args.output.expanduser().resolve()

    if not input_path.is_file():
        logger.error("Input file not found: %s", input_path)
        return 2

    channel_names = None
    if args.channels is not None:
        try:
            channel_names = parse_channel_list(args.channels)
        except ValueError as exc:
            logger.error("%s", exc)
            return 2

    embedding_path = args.embedding_path.expanduser().resolve()
    if not embedding_path.is_dir():
        logger.error("Embedding directory not found: %s", embedding_path)
        return 2

    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Processing %s -> %s", input_path, output_dir)
    try:
        outputs = run_full_pipeline(
            input_path,
            outdir=output_dir,
            scene_index=args.scene_index,
            embedding_path=embedding_path,
            preprocess_workers=args.preprocess_workers,
            segment_workers=args.segment_workers,
            tissue_labels_mode=args.tissue_labels,
            channel_names=channel_names,
            channels_from_filename=args.channels_from_filename,
            dpn_checkpoint=args.dpn_checkpoint,
        )
    except Exception:
        logger.exception("Full pipeline failed for %s", input_path)
        return 1

    for name, path in outputs.items():
        logger.info("%s: %s", name, path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
