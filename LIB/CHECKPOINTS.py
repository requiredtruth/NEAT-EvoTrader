"""Atomic, every-generation checkpoints and separate global-best saves."""
from __future__ import annotations
import csv
import json
import os
import pickle
from pathlib import Path


def atomic_pickle(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    with temp.open("wb") as handle: pickle.dump(value, handle)
    os.replace(temp, path)


def save_generation(run_dir: Path, state: dict) -> Path:
    path = run_dir / "CHECKPOINTS" / f"GEN_{state['generation']:06d}.pkl"
    atomic_pickle(path, state)
    history = state["history"]
    with (run_dir / "history.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=sorted(history[0]) if history else ["generation"])
        writer.writeheader(); writer.writerows(history)
    return path


def load(path: str | Path) -> dict:
    with Path(path).open("rb") as handle: return pickle.load(handle)


def save_best(best_dir: Path, genome, summary: dict) -> None:
    atomic_pickle(best_dir / "best_genome.pkl", genome)
    best_dir.mkdir(parents=True, exist_ok=True)
    (best_dir / "best_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")

