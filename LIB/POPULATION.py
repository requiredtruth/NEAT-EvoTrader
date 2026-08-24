"""Small, reproducible NEAT-style evolutionary loop."""
from __future__ import annotations
import json
import random
from pathlib import Path
from .CHECKPOINTS import save_best, save_generation
from .CROSSOVER import crossover
from .FITNESS import score
from .GENOME import Genome, InnovationTracker
from .MUTATION import mutate
from .SIMULATOR import run


def train(features, prices, config: dict, run_dir: Path, resume: dict | None = None) -> dict:
    rng = random.Random(config["seed"])
    inputs, outputs = features.shape[1] + 8, 8
    if resume:
        population, tracker, history, start = resume["population"], resume["tracker"], resume["history"], resume["generation"] + 1
        rng.setstate(resume["rng_state"])
    else:
        population = [Genome.minimal(i, inputs, outputs, rng) for i in range(config["population"])]
        tracker, history, start = InnovationTracker(inputs * outputs + 1), [], 1
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "config_snapshot.json").write_text(json.dumps(config, indent=2, sort_keys=True) + "\n")
    best = None
    for generation in range(start, config["generations"] + 1):
        results = []
        for genome in population:
            result = run(genome, features, prices, initial_cash=config["initial_cash"], max_slots=config["max_long_positions"], fee_rate=config["fee_rate"], slippage_rate=config["slippage_rate"], ladder=config["size_ladder"])
            genome.fitness = score(result, config["initial_cash"]); results.append(result)
        ranked = sorted(zip(population, results), key=lambda pair: pair[0].fitness or -1e99, reverse=True)
        best, best_result = ranked[0]
        history.append({"generation": generation, "best_fitness": best.fitness, "mean_fitness": sum(g.fitness or 0 for g in population) / len(population), "final_equity": best_result.final_equity, "max_drawdown": best_result.max_drawdown, "trades": best_result.trades, "nodes": len(best.biases), "connections": len(best.connections)})
        elites = [g.clone() for g, _ in ranked[:config["elitism"]]]
        children = elites[:]
        while len(children) < config["population"]:
            a, b = rng.choice(elites), rng.choice(elites)
            child = crossover(a, b, generation * 100000 + len(children), rng)
            roll = rng.random(); regime = "subtle" if roll < .72 else "heavy" if roll < .95 else "jump"
            mutate(child, rng, tracker, regime); children.append(child)
        state = {"generation": generation, "population": children, "tracker": tracker, "rng_state": rng.getstate(), "history": history, "config": config}
        checkpoint = save_generation(run_dir, state)
        save_best(Path("BEST"), best, {**history[-1], "checkpoint": str(checkpoint)})
        population = children
    return {"best": best, "history": history, "checkpoint": str(checkpoint)}

