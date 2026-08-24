"""Innovation-aligned NEAT crossover."""
from __future__ import annotations
import random
from .GENOME import Genome


def crossover(a: Genome, b: Genome, key: int, rng: random.Random) -> Genome:
    fitter, other = (a, b) if (a.fitness or -1e99) >= (b.fitness or -1e99) else (b, a)
    child = fitter.clone(key)
    for innovation, gene in list(child.connections.items()):
        if innovation in other.connections and rng.random() < 0.5:
            child.connections[innovation] = other.clone().connections[innovation]
    for node in child.biases:
        if node in other.biases and rng.random() < 0.5:
            child.biases[node] = other.biases[node]
    return child

