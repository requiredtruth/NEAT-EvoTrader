# NEAT-EvoTrader

CPU-first topology evolution for deterministic research over local CSV OHLC data. It evolves feed-forward policies that can independently open and close fixed long and short position slots, including simultaneous long and short exposure. It has no broker, exchange, wallet, credential, network-data, or live-order interface.

> Research software, not financial advice. Historical and synthetic results do not predict future results. Do not deploy evolved policies with real funds without independent validation.

## Verify it in one command

```console
$ ./doit.sh
...
Ran 5 tests
OK
generation=2 best_fitness=...
checkpoint=...
```

The command runs the complete test suite and a seeded two-generation synthetic-data demonstration. Normal numerical results vary across Python/NumPy versions; the run is deterministic within the same environment and seed.

## What it implements

- strict, local-only CSV ingestion (`open,high,low,close`, optional `volume`)
- fixed-width causal features and account/action memory
- three independent long slots and three independent short slots by default
- new positions instead of averaging; leverage decoded and clamped to 2–100
- fees, slippage, deterministic final liquidation, invalid-action penalties, inactivity penalties, and drawdown-aware fitness
- feed-forward acyclic genomes with innovation-aligned crossover
- subtle, heavy, and rare topology-jump mutation regimes
- atomic checkpoint after every generation, including population, innovation tracker, RNG state, history, and configuration
- separately saved global-best genome and JSON summary
- NumPy array simulation; Numba is an optional installed accelerator target for later compiled hot paths

## Run a research job

```bash
python3 -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
python main.py DATA/sample_ohlc.csv --generations 20 --population 48 --seed 7
```

Resume from the exact stored evolutionary state:

```bash
python main.py DATA/sample_ohlc.csv --generations 40 --resume RUNS/<run>/CHECKPOINTS/GEN_000020.pkl --run-dir RUNS/<run>
```

Common validation errors are deliberately exact:

```text
missing required CSV columns: high, low
non-numeric OHLC value on CSV line 17
invalid OHLC relationship on CSV line 9
at least 32 OHLC rows are required
```

## Scope and comparison

This is not a replacement for general NEAT libraries, vectorized portfolio research suites, or reinforcement-learning platforms. The narrower goal is an inspectable baseline joining topology growth, a side-aware multi-slot simulator, and exact evolutionary resume in one local historical-replay tool. See [neat-python](https://github.com/CodeReclaimers/neat-python), [VectorBT](https://github.com/polakowo/vectorbt), and [FinRL](https://github.com/AI4Finance-Foundation/FinRL) for those broader use cases.

## Honest limitations

- Version 0.1 evaluates one symbol and timeframe per run.
- Species protection, multiprocessing, the Tkinter/Matplotlib dashboard, and a compiled Numba evaluator are planned but not yet implemented.
- Checkpoints use Python pickle. Load only checkpoints you created and trust.
- The included OHLC file is synthetic demonstration data, not market history.
- The simulator is intentionally simplified and cannot model liquidity, liquidation, funding, taxes, or execution uncertainty.

## Support

Public donation addresses and the self-service confirmed-transaction request process are in [SUPPORT.md](SUPPORT.md). Confirm the asset and network before sending.

## License

MIT. See [LICENSE](LICENSE).
