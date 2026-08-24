"""Subtle, heavy, and rare structural-jump mutation regimes."""
from __future__ import annotations
import random
from .GENOME import Connection, Genome, InnovationTracker


def mutate(genome: Genome, rng: random.Random, tracker: InnovationTracker, regime: str = "subtle") -> str:
    if regime not in {"subtle", "heavy", "jump"}:
        raise ValueError("regime must be subtle, heavy, or jump")
    scale = {"subtle": 0.12, "heavy": 0.8, "jump": 2.0}[regime]
    probability = {"subtle": 0.25, "heavy": 0.65, "jump": 1.0}[regime]
    for connection in genome.connections.values():
        if rng.random() < probability:
            connection.weight += rng.gauss(0, scale)
    for node in genome.biases:
        if rng.random() < probability:
            genome.biases[node] += rng.gauss(0, scale)
    if regime in {"heavy", "jump"} and genome.connections:
        split = rng.choice(list(genome.connections.values()))
        if split.enabled:
            split.enabled = False
            node = max([genome.inputs + genome.outputs - 1, *genome.biases]) + 1
            genome.biases[node] = 0.0
            for source, target, weight in ((split.source, node, 1.0), (node, split.target, split.weight)):
                innovation = tracker.take()
                genome.connections[innovation] = Connection(source, target, weight, innovation)
    return regime

