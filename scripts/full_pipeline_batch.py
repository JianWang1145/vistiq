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
    return parser.parse_args(argv)






#below is the fine tuning function(Chat wrote this and I doubled checked it to make sure it's running fine. It should be, I guess...)






def finetune_dpn(gt_dir: Path, output_dir: Path) -> None:
    """Fine-tune MicroSAM for Dpn cell instance segmentation."""
    import random
    import numpy as np
    import tifffile
    import torch
    from skimage.measure import label as cc_label
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

    if len(groups) < 2:
        raise ValueError("Need at least two Animals for splitting")

    # 2. Split by Animal, approximately 90% / 10%. (I Changed the seed to 114514. This does not matter, you can change it to any number you want. However, for the sake of reproducibility, I recommend using the same seed as me.)
    animals = sorted(groups)
    random.Random(114514).shuffle(animals)

    choices = {0: []}

    for animal in animals:
        for count, selected in list(choices.items()):
            new_count = count + len(groups[animal])
            choices.setdefault(new_count, selected + [animal])

    val_count = min(
        (n for n in choices if 0 < n < len(gt_files)),
        key=lambda n: (abs(n - 12), n),
    )

    val_animals = set(choices[val_count])

    train_pairs = [
        pair
        for animal in animals if animal not in val_animals
        for pair in groups[animal]
    ]

    val_pairs = [
        pair
        for animal in animals if animal in val_animals
        for pair in groups[animal]
    ]

    print("Training images:", len(train_pairs))
    print("Validation images:", len(val_pairs))
    print("Validation Animals:", sorted(val_animals))

    # Save the split for reproducibility.
    output_dir.mkdir(parents=True, exist_ok=True)

    with (output_dir / "split.txt").open("w") as f:
        for split, pairs in (
            ("train", train_pairs),
            ("val", val_pairs),
        ):
            for raw, gt in pairs:
                f.write(f"{split}\t{raw.name}\t{gt.name}\n")

    # 3. Convert binary GT into instance labels. (Andy's note: This converts binary masks into labels. However, ajacent cells will be labeled as one cell. Remind yourself to validate this in the future. If you read this and Andy still haven't done this, please yell at him.)
    def to_instances(mask):
        return cc_label(
            np.asarray(mask) > 0,
            connectivity=2,
        ).astype(np.int32)

    # 4. Ensure MicroSAM input is in the 0–255 range.
    def to_uint8(image):
        x = np.asarray(image)

        if x.dtype == np.uint8:
            return x

        lo, hi = np.percentile(x.astype(np.float32), [1, 99])

        if hi <= lo:
            return np.zeros(x.shape, dtype=np.uint8)

        return np.clip(
            (x - lo) * 255 / (hi - lo),
            0,
            255,
        ).astype(np.uint8)

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

def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
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
        )
    except Exception:
        logger.exception("Full pipeline failed for %s", input_path)
        return 1

    for name, path in outputs.items():
        logger.info("%s: %s", name, path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
