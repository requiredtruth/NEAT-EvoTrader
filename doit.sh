#!/usr/bin/env sh
set -eu
export MPLCONFIGDIR="${TMPDIR:-/tmp}/neat-evotrader-matplotlib"
python3 -m unittest discover -s tests -v
python3 main.py --generations 2 --population 8 --seed 7 --run-dir "${TMPDIR:-/tmp}/neat-evotrader-demo"

