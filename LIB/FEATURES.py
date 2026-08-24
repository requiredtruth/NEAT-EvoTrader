"""Fixed-width, causal market features."""
from __future__ import annotations
import numpy as np


def build_features(ohlcv: np.ndarray) -> np.ndarray:
    """Build causal features; each row only uses current and earlier bars."""
    o, h, l, c, v = ohlcv.T
    out = np.zeros((len(c), 12), dtype=np.float64)
    prev = np.r_[c[0], c[:-1]]
    out[:, 0] = (c / prev) - 1.0
    out[:, 1] = (h - l) / c
    out[:, 2] = (c - o) / c
    out[:, 3] = (c - l) / np.maximum(h - l, 1e-12)
    out[:, 4] = (h - np.maximum(o, c)) / c
    out[:, 5] = (np.minimum(o, c) - l) / c
    for col, window in enumerate((3, 8, 21), 6):
        for i in range(window - 1, len(c)):
            segment = c[i - window + 1 : i + 1]
            out[i, col] = c[i] / segment.mean() - 1.0
            if col == 8:
                out[i, 9] = np.std(np.diff(segment) / segment[:-1])
    out[:, 10] = np.log1p(v) / max(float(np.log1p(v).max()), 1.0)
    out[:, 11] = np.arange(len(c), dtype=float) / max(len(c) - 1, 1)
    return np.nan_to_num(out)

