import json
import random
import tempfile
import unittest
from pathlib import Path
import numpy as np

from LIB.CHECKPOINTS import load
from LIB.DATASET import load_ohlc
from LIB.FEATURES import build_features
from LIB.GENOME import Genome, InnovationTracker
from LIB.MUTATION import mutate
from LIB.POPULATION import train
from LIB.SIMULATOR import decode, run
from gui import run_demo


ROOT = Path(__file__).parents[1]


class FixedPolicy:
    def activate(self, state):
        return np.asarray([.9, .9, -.9, -.9, 0, 0, -.8, -.8])


class EngineTests(unittest.TestCase):
    def test_loader_and_features_are_finite(self):
        data = load_ohlc(ROOT / "DATA/sample_ohlc.csv")
        features = build_features(data)
        self.assertEqual(features.shape, (40, 12))
        self.assertTrue(np.isfinite(features).all())

    def test_simultaneous_long_and_short_books(self):
        data = load_ohlc(ROOT / "DATA/sample_ohlc.csv")
        result = run(FixedPolicy(), build_features(data), data[:, 3])
        self.assertGreater(result.long_opens, 0)
        self.assertGreater(result.short_opens, 0)
        self.assertGreater(result.trades, 0)
        self.assertAlmostEqual(result.final_equity, result.realized_pnl + 10000.0)

    def test_decoder_bounds_leverage(self):
        values = decode(np.asarray([0, 0, 0, 0, -99, 99, -99, 99]), (.01, .1))
        self.assertEqual(values[4:6], (2, 100))
        self.assertEqual(values[6:], (.01, .1))

    def test_heavy_mutation_grows_topology(self):
        rng = random.Random(4)
        genome = Genome.minimal(1, 3, 2, rng)
        nodes = len(genome.biases)
        mutate(genome, rng, InnovationTracker(), "heavy")
        self.assertGreater(len(genome.biases), nodes)

    def test_checkpoint_is_loadable_and_resumable(self):
        data = load_ohlc(ROOT / "DATA/sample_ohlc.csv"); features = build_features(data)
        config = json.loads((ROOT / "CONFIG/DEFAULTS.json").read_text())
        config.update(population=6, generations=1, seed=3)
        with tempfile.TemporaryDirectory() as tmp:
            best_dir = Path(tmp) / "best"
            first = train(features, data[:, 3], config, Path(tmp) / "one", best_dir=best_dir)
            state = load(first["checkpoint"])
            self.assertEqual(state["generation"], 1)
            config["generations"] = 2
            second = train(features, data[:, 3], config, Path(tmp) / "two", state, best_dir)
            self.assertEqual(second["history"][-1]["generation"], 2)

    def test_safe_demo_does_not_modify_durable_best(self):
        def snapshot():
            return {
                str(path.relative_to(ROOT / "BEST")): path.read_bytes()
                for path in (ROOT / "BEST").rglob("*")
                if path.is_file()
            }

        before = snapshot()
        result = run_demo()
        self.assertEqual(result["status"], "PASS")
        self.assertTrue(result["artifacts_removed"])
        self.assertIn("<temporary-demo>", result["output"])
        self.assertEqual(snapshot(), before)


if __name__ == "__main__": unittest.main()
