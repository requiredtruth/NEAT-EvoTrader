#!/usr/bin/env python3
"""NEAT-EvoTrader command line."""
from __future__ import annotations
import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from LIB.CHECKPOINTS import load
from LIB.DATASET import load_ohlc
from LIB.FEATURES import build_features
from LIB.POPULATION import train


def main() -> int:
    parser = argparse.ArgumentParser(description="CPU-first NEAT research over historical OHLC CSV files")
    parser.add_argument("csv", nargs="?", default="DATA/sample_ohlc.csv")
    parser.add_argument("--config", default="CONFIG/DEFAULTS.json")
    parser.add_argument("--generations", type=int)
    parser.add_argument("--population", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--resume")
    parser.add_argument("--run-dir")
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text())
    for key in ("generations", "population", "seed"):
        if getattr(args, key) is not None: config[key] = getattr(args, key)
    if config["max_long_positions"] != config["max_short_positions"]:
        raise SystemExit("current engine requires equal long and short slot counts")
    data = load_ohlc(args.csv); features = build_features(data)
    run_dir = Path(args.run_dir or f"RUNS/{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}")
    outcome = train(features, data[:, 3], config, run_dir, load(args.resume) if args.resume else None)
    last = outcome["history"][-1]
    print(f"generation={last['generation']} best_fitness={last['best_fitness']:.6f} equity={last['final_equity']:.2f} trades={last['trades']}")
    print(f"checkpoint={outcome['checkpoint']}")
    return 0


if __name__ == "__main__": raise SystemExit(main())

