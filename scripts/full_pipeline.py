#!/usr/bin/env python3
"""Full hierarchical segmentation and analysis (notebook: hierarchical-analysis-gwas.ipynb).

For each ``.lif`` volume under an input directory: preprocess and segment tissue
(lobes + brain mask), segment cell channels, run hierarchical overlap/spatial
analysis, and write label TIFFs plus measurement CSVs next to the input file.
Failed files are skipped; a summary is logged at the end.

Intended for batch use via scripts/batch_process.sbatch (one file per array task)::

    python scripts/full_pipeline.py -i /path/to/data/folder

Or process every ``*.lif`` under a tree in one invocation::

    python scripts/full_pipeline.py -i /path/to/data/folder --recursive
"""

from __future__ import annotations

import argparse
import copy
import logging
import os
import sys
from pathlib import Path
from typing import Any, Literal

import numpy as np
from prefect import flow
from scipy.ndimage import binary_dilation

import vistiq
from vistiq.analysis import (
    AnalysisFlow,
    AnalysisFlowConfig,
    IoSMetricsCalculatorConfig,
    KnnAnalysisConfig,
    LabelOverlapCalculatorConfig,
    RnnAnalysisConfig,
    SpatialScopeConfig,
)
from vistiq.graph import HierarchyBuilderConfig
from vistiq.matrix.ops import MatrixAggregatorConfig
from vistiq.matrix.types import FULL
from vistiq.io import (
    DataFrameWriter,
    DataFrameWriterConfig,
    FileList,
    FileListConfig,
    ImageLoader,
    ImageLoaderConfig,
    ImageWriter,
    ImageWriterConfig,
    unstack_image,
)
from vistiq.preprocess import (
    FuncProcessor,
    FuncProcessorConfig,
    PreprocessFlow,
    PreprocessFlowConfig,
    RescaleConfig,
)
from vistiq.segment import (
    MicroSAMSegmenterConfig,
    RangeFilterConfig,
    RegionAnalyzerConfig,
    RegionFilterConfig,
    SegmentationFlow,
    SegmentationFlowConfig,
    TiledSegmentationFlow,
    TiledSegmentationFlowConfig,
)
from vistiq.segment.select import ValueFilterConfig
from vistiq.utils import ArrayIteratorConfig, check_device, resolve_futures

TissueLabelsMode = Literal["2d", "3d"]

logger = logging.getLogger(__name__)

DEFAULT_RENAME_CHANNEL = {"Red": "Dpn", "Green": "InR", "Blue": "EdU"}
DEFAULT_EMBEDDING_PATH = os.environ.get("VISTIQ_EMBEDDING_PATH", "./embeddings")
LIF_PATTERN = "*.lif"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run tissue + cell segmentation and hierarchical analysis on "
            "microscopy volumes in an input directory."
        ),
    )
    parser.add_argument(
        "-i",
        "--input",
        required=True,
        type=Path,
        help="Input directory to search for .lif files",
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
        "--recursive",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Search input directory recursively for .lif files (default: true)",
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
            "lobe labels to 3D (GWAS notebook default); 3d segments the full volume "
            "and uses stricter cross-sectional area filters on all planes (default: 2d)"
        ),
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level (default: INFO)",
    )
    return parser.parse_args(argv)


def parse_channel_list(value: str) -> list[str]:
    names = [part.strip() for part in value.split(",") if part.strip()]
    if not names:
        raise ValueError("Expected at least one channel name in --channels")
    return names


def channel_names_from_filename(path: Path, n_channels: int) -> list[str] | None:
    if n_channels < 1:
        return None
    stem = path.name.rsplit(".", 1)[0]
    parts = [part.strip() for part in stem.split("-") if part.strip()]
    if len(parts) < n_channels:
        return None
    return parts[-n_channels:]


def resolve_channel_names(
    metadata: dict[str, Any],
    input_path: Path,
    *,
    channel_names: list[str] | None,
    channels_from_filename: bool,
) -> None:
    loader_names = metadata.get("channel_names")
    if not loader_names:
        return

    if channel_names is not None:
        if len(channel_names) != len(loader_names):
            raise ValueError(
                f"--channels provided {len(channel_names)} name(s) but image has "
                f"{len(loader_names)} channel(s): {loader_names}"
            )
        metadata["channel_names"] = channel_names
        logger.info("Channel names from --channels: %s", channel_names)
        return

    if not channels_from_filename:
        return

    parsed = channel_names_from_filename(input_path, len(loader_names))
    if parsed is not None:
        metadata["channel_names"] = parsed
        logger.info("Channel names from filename: %s", parsed)
        return

    metadata["channel_names"] = [
        DEFAULT_RENAME_CHANNEL.get(name, name) for name in loader_names
    ]
    logger.warning(
        "Could not parse %d channel name(s) from %r; using DEFAULT_RENAME_CHANNEL",
        len(loader_names),
        input_path.name,
    )


def discover_image_files(
    input_dir: Path,
    *,
    recursive: bool,
    include: str = LIF_PATTERN,
) -> list[Path]:
    """Find image files under *input_dir* matching *include* using :class:`FileList`."""
    input_dir = input_dir.expanduser().resolve()
    if not input_dir.is_dir():
        raise NotADirectoryError(f"Input path is not a directory: {input_dir}")

    files = FileList(
        FileListConfig(
            paths=input_dir,
            include=include,
            recursive=recursive,
        )
    ).run()
    return [path.resolve() for path in files if path.is_file()]


def _base_tissue_preprocessors() -> list:
    """Shared tissue preprocessors from hierarchical-analysis-gwas.ipynb."""
    return [
        RescaleConfig(
            low=2,
            high=98,
            dtype=np.uint8,
            iterator_config=ArrayIteratorConfig(slice_def=(-3, -2, -1)),
        ),
        FuncProcessorConfig(
            func="skimage.filters.gaussian",
            kwargs={"sigma": 1.0},
            iterator_config=ArrayIteratorConfig(slice_def=(-2, -1)),
        ),
        FuncProcessorConfig(
            func="skimage.exposure.adjust_gamma",
            kwargs={"gamma": 0.2},
            iterator_config=ArrayIteratorConfig(slice_def=(-3, -2, -1)),
        ),
        FuncProcessorConfig(
            func="skimage.exposure.adjust_sigmoid",
            iterator_config=ArrayIteratorConfig(slice_def=(-3, -2, -1)),
        ),
        RescaleConfig(
            dtype=np.uint8,
            iterator_config=ArrayIteratorConfig(slice_def=(-3, -2, -1)),
        ),
        FuncProcessorConfig(
            func="numpy.max",
            kwargs={"axis": ("C",)},
            strict_axis=False,
            dtype=np.uint16,
        ),
    ]


def build_tissue_preprocess_config(mode: TissueLabelsMode) -> PreprocessFlowConfig:
    """Build tissue preprocess config for 2D or 3D lobe segmentation."""
    processors = _base_tissue_preprocessors()
    if mode == "2d":
        processors.append(
            FuncProcessorConfig(
                func="numpy.sum",
                kwargs={"axis": ("Z",)},
                strict_axis=False,
            )
        )
    return PreprocessFlowConfig(processors=processors)


def build_tissue_region_filter_config(mode: TissueLabelsMode) -> RegionFilterConfig:
    """Build tissue region filters for 2D or 3D lobe segmentation."""
    filters = [
        RangeFilterConfig(
            attribute="cross_sectional_area-xy",
            range=(2000, np.inf),
        ),
    ]
    if mode == "3d":
        filters.extend(
            [
                RangeFilterConfig(
                    attribute="cross_sectional_area-xz",
                    range=(2000, np.inf),
                ),
                RangeFilterConfig(
                    attribute="cross_sectional_area-yz",
                    range=(2000, np.inf),
                ),
            ]
        )
    filters.append(
        RangeFilterConfig(
            attribute="aspect_ratio",
            range=(0.5, 1.0),
        )
    )
    return RegionFilterConfig(filters=filters)


def build_cell_preprocess_config() -> PreprocessFlowConfig:
    return PreprocessFlowConfig(processors=[])


def build_tissue_segmentation_config(
    embedding_path: Path,
    mode: TissueLabelsMode,
) -> TiledSegmentationFlowConfig:
    return TiledSegmentationFlowConfig(
        segmenter=MicroSAMSegmenterConfig(
            iterator_config=ArrayIteratorConfig(slice_def=()),
            embedding_path=str(embedding_path),
        ),
        region_filter=build_tissue_region_filter_config(mode),
        tile_factor=(3, 3),
        resize_factor=(0.25, 0.25),
        iou_threshold=0.5,
        consensus_threshold=0.75,
    )


def build_cell_segmentation_config(embedding_path: Pathc, heckpoint: Path | None = None,) -> SegmentationFlowConfig:
    min_cell_radius = 2.0
    max_cell_radius = 7.0
    return SegmentationFlowConfig(
        segmenter=MicroSAMSegmenterConfig(
            model_type="vit_l" if checkpoint else "vit_l_lm",
            checkpoint=str(checkpoint) if checkpoint else None,
            iterator_config=ArrayIteratorConfig(slice_def=()),
            embedding_path=str(embedding_path),
        ),
        region_filter=RegionFilterConfig(
            filters=[
                RangeFilterConfig(
                    attribute="cross_sectional_area-xy",
                    range=(np.pi * min_cell_radius**2, np.pi * max_cell_radius**2),
                )
            ]
        ),
    )


def build_analysis_config() -> AnalysisFlowConfig:
    return AnalysisFlowConfig(
        region_analyzer=RegionAnalyzerConfig(
            properties=[
                "volume",
                "centroid",
                "cross_sectional_area",
                "bbox",
                "aspect_ratio",
            ],
            iterator_config=ArrayIteratorConfig(slice_def=()),
            output_type="dataframe",
            index_on="object_id",
            map_axes=True,
        ),
        hierarchy_builder=HierarchyBuilderConfig(orphan_strategy="drop"),
        overlap_calculator=LabelOverlapCalculatorConfig(
            metrics_calculators=[IoSMetricsCalculatorConfig()],
            output_type="dataframe",
            annotate=True,
            triangle=FULL,
        ),
        overlap_filter=ValueFilterConfig(
            ref_value=0.5,
            axis=0,
            operator=">",
            triangle=FULL,
            output="masked_values",
        ),
        overlap_aggregator=MatrixAggregatorConfig(
            operation="count",
            axis=1,
        ),
        knn_analysis=KnnAnalysisConfig(
            k=5,
            mode="homotypic",
            scope=SpatialScopeConfig(match={"channel": "Lobe"}),
        ),
        rnn_analysis=RnnAnalysisConfig(
            radius=25,
            mode="homotypic",
            scope=SpatialScopeConfig(match={"channel": "Lobe"}),
        ),
    )


def expand_2d_tissue_labels(
    tissue_labels: np.ndarray,
    tissue_metadata: dict[str, Any],
    img: np.ndarray,
) -> tuple[np.ndarray, dict[str, Any]]:
    logger.info("Expanding 2D tissue mask to 3D (Z=%d)", img.shape[-3])
    fc = FuncProcessorConfig(
        func="numpy.repeat",
        args=[img.shape[-3]],
        kwargs={"axis": (0,)},
        strict_axis=False,
        output_dims={"Z": img.shape[-3], "Y": img.shape[-2], "X": img.shape[-1]},
        dtype=np.uint16,
    )
    arr = tissue_labels[np.newaxis, :, :]
    tissue_labels, tissue_metadata = FuncProcessor(fc).run(arr, metadata=tissue_metadata)
    n_labels = len(np.unique(tissue_labels)) - 1
    if n_labels > 0:
        scale = np.max(tissue_labels) // n_labels
        if scale > 0:
            tissue_labels = tissue_labels // scale
    return tissue_labels, tissue_metadata


def build_organ_label(tissue_labels: np.ndarray) -> np.ndarray:
    brain_mask = (tissue_labels > 0).astype(np.uint16)
    for i in range(len(brain_mask)):
        brain_mask[i] = binary_dilation(brain_mask[i], iterations=1)
    return (brain_mask * 1).astype(np.uint16)


def save_outputs(
    combined_labels: list[np.ndarray],
    combined_metadata: list[dict[str, Any]],
    measurements: dict[str, Any],
    *,
    img_path: Path,
    outdir: Path,
) -> dict[str, Path]:
    outdir.mkdir(parents=True, exist_ok=True)
    fname_stem = img_path.stem
    scene = combined_metadata[0].get("scene_index", "")
    base = outdir / f"{fname_stem}.scene-{scene}"

    writer_config = ImageWriterConfig(overwrite=True)
    outpaths = [
        outdir / f"{fname_stem}.scene-{meta.get('scene_index', '')}.tif"
        for meta in combined_metadata
    ]
    ImageWriter(writer_config).run.map(
        combined_labels,
        [str(path) for path in outpaths],
        metadata=combined_metadata,
    )

    dw = DataFrameWriter(
        DataFrameWriterConfig(format="csv", overwrite=True, save_index=True)
    )
    region_csv = base.with_suffix(".region_analyzer_all.csv")
    ios_csv = base.with_suffix(".ios_global.csv")
    ios_ann_csv = Path(f"{base}.ios_global.annotated.csv")

    dw.run(measurements["region_analyzer_all"], region_csv)
    dw.run(measurements["ios_global"], ios_csv)

    regions = measurements["region_analyzer_all"]
    name_map = regions["object_name"]
    if name_map.index.name != "object_id" and "object_id" in regions.columns:
        name_map = regions.set_index("object_id")["object_name"]
    ios_ann = measurements["ios_global"].rename(index=name_map, columns=name_map)
    dw.run(ios_ann, ios_ann_csv)

    label_tifs = {
        meta["channel_names"][0]: path.with_suffix(
            f".{meta['channel_names'][0]}.tif"
        )
        for path, meta in zip(outpaths, combined_metadata)
    }
    return {
        "region_analyzer_all": region_csv,
        "ios_global": ios_csv,
        "ios_global_annotated": ios_ann_csv,
        **{f"labels_{name}": tif for name, tif in label_tifs.items()},
    }


@flow(name="vistiq.full_pipeline")
def run_full_pipeline(
    img_path: Path,
    *,
    outdir: Path | None = None,
    scene_index: int = 0,
    embedding_path: Path = Path(DEFAULT_EMBEDDING_PATH),
    preprocess_workers: int = -1,
    segment_workers: int = 2,
    tissue_labels_mode: TissueLabelsMode = "2d",
    channel_names: list[str] | None = None,
    channels_from_filename: bool = False,
    dpn_checkpoint: Path | None = None,
) -> dict[str, Path]:
    """Run the GWAS notebook pipeline for a single image file."""
    img_path = img_path.expanduser().resolve()
    embedding_path = embedding_path.expanduser().resolve()
    if outdir is None:
        outdir = img_path.parent
    else:
        outdir = outdir.expanduser().resolve()

    if not img_path.is_file():
        raise FileNotFoundError(f"Input file not found: {img_path}")
    if not embedding_path.is_dir():
        raise FileNotFoundError(f"Embedding directory not found: {embedding_path}")

    logger.info("Loading %s (scene_index=%s)", img_path, scene_index)
    loader_config = ImageLoaderConfig(
        squeeze=True,
        rename_channel=None if channel_names or channels_from_filename else DEFAULT_RENAME_CHANNEL,
        scene_index=scene_index,
        split_channels=False,
    )
    img, metadata = ImageLoader(loader_config).run(img_path)
    resolve_channel_names(
        metadata,
        img_path,
        channel_names=channel_names,
        channels_from_filename=channels_from_filename,
    )

    logger.info("Tissue labels mode: %s", tissue_labels_mode)
    tissue_ppcfg = build_tissue_preprocess_config(tissue_labels_mode)
    tsfcfg = build_tissue_segmentation_config(embedding_path, tissue_labels_mode)
    cell_ppcfg = build_cell_preprocess_config()
    acfg = build_analysis_config()

    logger.info("Preprocessing tissue")
    tissue_img, tissue_metadata = PreprocessFlow(tissue_ppcfg).run(
        img,
        metadata=metadata,
        workers=preprocess_workers,
    )

    logger.info("Segmenting tissue")
    tissue_labels = TiledSegmentationFlow(tsfcfg).run(
        tissue_img,
        metadata=tissue_metadata,
        workers=segment_workers,
        verbose=0,
    )
    if tissue_labels_mode == "2d":
        if tissue_labels.ndim == img.ndim - 2:
            tissue_labels, tissue_metadata = expand_2d_tissue_labels(
                tissue_labels,
                tissue_metadata,
                img,
            )
        else:
            logger.warning(
                "2D tissue mode but segmentation returned ndim=%s (expected %s)",
                tissue_labels.ndim,
                img.ndim - 2,
            )
    elif tissue_labels.ndim == img.ndim - 2:
        logger.warning(
            "3D tissue mode but segmentation returned 2D labels; expanding to volume"
        )
        tissue_labels, tissue_metadata = expand_2d_tissue_labels(
            tissue_labels,
            tissue_metadata,
            img,
        )
    else:
        logger.info("Using 3D tissue mask")

    tissue_metadata = copy.deepcopy(tissue_metadata)
    tissue_metadata["channel_names"] = ["Lobe"]

    logger.info("Building brain mask")
    brain_label = build_organ_label(tissue_labels)
    brain_metadata = copy.deepcopy(tissue_metadata)
    brain_metadata["channel_names"] = ["Brain"]

    logger.info("Preprocessing and segmenting cell channels")
    preprocessed, preprocessed_metadata = PreprocessFlow(cell_ppcfg).run(
        img,
        metadata=metadata,
    )
    channels, channel_metadata = unstack_image(
        preprocessed,
        preprocessed_metadata,
        axis=metadata["channel_axis"],
        strict=False,
    )
    cell_labels = []
    dpn_found = False

    for ch, ch_meta in zip(channels, channel_metadata):

        channel_name = ch_meta["channel_names"][0]
        is_dpn = channel_name.lower() == "dpn"

        if is_dpn:
            dpn_found = True

        if is_dpn and dpn_checkpoint is not None:
            # Use fine-tuned model only for Dpn.
            config = build_cell_segmentation_config(
                embedding_path / "dpn_finetuned",
                checkpoint=dpn_checkpoint,
            )
            logger.info(
                "Segmenting %s with fine-tuned model: %s",
                channel_name,
                dpn_checkpoint,
            )
        else:
            # All other channels keep the original model.
            config = build_cell_segmentation_config(
                embedding_path,
            )
            logger.info(
                "Segmenting %s with pretrained MicroSAM",
                channel_name,
            )

        labels = SegmentationFlow(config).run(
            ch,
            metadata=ch_meta,
        )

        cell_labels.append(labels)

    if dpn_checkpoint is not None and not dpn_found:
        raise ValueError(
            "A Dpn checkpoint was provided, but no Dpn channel was found."
        )


    combined_labels = [brain_label, tissue_labels, *cell_labels]
    combined_metadata = [brain_metadata, tissue_metadata, *channel_metadata]

    logger.info("Running hierarchical analysis")
    measurements = AnalysisFlow(acfg).run(combined_labels, metadata=combined_metadata)
    measurements = resolve_futures(measurements)

    outputs = save_outputs(
        combined_labels,
        combined_metadata,
        measurements,
        img_path=img_path,
        outdir=outdir,
    )
    logger.info("Wrote outputs for %s to %s", img_path.name, outdir)
    return outputs


def log_processing_summary(
    successful: list[tuple[Path, dict[str, Path]]],
    failed: list[tuple[Path, str]],
) -> None:
    """Log a batch summary of successful and failed input files."""
    logger.info("=" * 60)
    logger.info(
        "Processing summary: %d succeeded, %d failed, %d total",
        len(successful),
        len(failed),
        len(successful) + len(failed),
    )
    if successful:
        logger.info("Successful (%d):", len(successful))
        for path, _outputs in successful:
            logger.info("  OK  %s", path)
    if failed:
        logger.error("Failed (%d):", len(failed))
        for path, error in failed:
            logger.error("  FAIL  %s: %s", path, error)
    logger.info("=" * 60)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    )
    logging.getLogger(vistiq.__name__).setLevel(getattr(logging, args.log_level))
    logger.info("Available Torch accelerators: %s", check_device())

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

    try:
        image_paths = discover_image_files(args.input, recursive=args.recursive)
    except NotADirectoryError as exc:
        logger.error("%s", exc)
        return 2

    if not image_paths:
        logger.warning("No .lif files found in %s", args.input)
        return 0

    logger.info(
        "Found %d .lif file(s) under %s (recursive=%s)",
        len(image_paths),
        args.input,
        args.recursive,
    )

    successful: list[tuple[Path, dict[str, Path]]] = []
    failed: list[tuple[Path, str]] = []
    for image_path in image_paths:
        logger.info("Processing %s", image_path)
        try:
            outputs = run_full_pipeline(
                image_path,
                scene_index=args.scene_index,
                embedding_path=embedding_path,
                preprocess_workers=args.preprocess_workers,
                segment_workers=args.segment_workers,
                tissue_labels_mode=args.tissue_labels,
                channel_names=channel_names,
                channels_from_filename=args.channels_from_filename,
            )
        except Exception as exc:
            logger.exception("Full pipeline failed for %s", image_path)
            failed.append((image_path, f"{type(exc).__name__}: {exc}"))
            continue

        successful.append((image_path, outputs))
        for name, path in outputs.items():
            logger.info("  %s: %s", name, path)

    log_processing_summary(successful, failed)

    if failed:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
