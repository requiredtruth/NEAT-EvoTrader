"""Strict CSV OHLC ingestion without network access."""
from __future__ import annotations

import csv
from pathlib import Path
import numpy as np

REQUIRED = ("open", "high", "low", "close")


def load_ohlc(path: str | Path) -> np.ndarray:
    """Return contiguous float64 columns: open, high, low, close, volume."""
    rows: list[list[float]] = []
    with Path(path).open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        names = {n.lower(): n for n in (reader.fieldnames or [])}
        missing = [n for n in REQUIRED if n not in names]
        if missing:
            raise ValueError("missing required CSV columns: " + ", ".join(missing))
        for line, row in enumerate(reader, 2):
            try:
                values = [float(row[names[n]]) for n in REQUIRED]
                volume = float(row[names["volume"]]) if "volume" in names and row[names["volume"]] else 0.0
            except (TypeError, ValueError) as exc:
                raise ValueError(f"non-numeric OHLC value on CSV line {line}") from exc
            o, h, l, c = values
            if min(values) <= 0 or h < max(o, c, l) or l > min(o, c, h):
                raise ValueError(f"invalid OHLC relationship on CSV line {line}")
            rows.append([o, h, l, c, volume])
    if len(rows) < 32:
        raise ValueError("at least 32 OHLC rows are required")
    return np.ascontiguousarray(rows, dtype=np.float64)

