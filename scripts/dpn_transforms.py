"""Reusable, pickle-compatible Dpn transforms."""

import numpy as np
from skimage.measure import label as cc_label


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

