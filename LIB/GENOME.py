"""Acyclic topology-growing genome and compiled feed-forward evaluator."""
from __future__ import annotations
from dataclasses import dataclass, field
import math
import random
import numpy as np


@dataclass
class Connection:
    source: int
    target: int
    weight: float
    innovation: int
    enabled: bool = True


@dataclass
class Genome:
    key: int
    inputs: int
    outputs: int
    biases: dict[int, float] = field(default_factory=dict)
    connections: dict[int, Connection] = field(default_factory=dict)
    fitness: float | None = None

    @classmethod
    def minimal(cls, key: int, inputs: int, outputs: int, rng: random.Random) -> "Genome":
        g = cls(key, inputs, outputs)
        for out in range(inputs, inputs + outputs):
            g.biases[out] = rng.uniform(-0.25, 0.25)
            for source in range(inputs):
                innovation = source * outputs + (out - inputs)
                g.connections[innovation] = Connection(source, out, rng.uniform(-1, 1), innovation)
        return g

    def clone(self, key: int | None = None) -> "Genome":
        import copy
        result = copy.deepcopy(self)
        result.key = self.key if key is None else key
        result.fitness = None
        return result

    def activate(self, state: np.ndarray) -> np.ndarray:
        values = {i: float(state[i]) for i in range(self.inputs)}
        nodes = sorted(self.biases)
        for node in nodes:
            total = self.biases[node]
            for c in self.connections.values():
                if c.enabled and c.target == node:
                    total += values.get(c.source, 0.0) * c.weight
            values[node] = math.tanh(max(-20.0, min(20.0, total)))
        return np.asarray([values.get(i, 0.0) for i in range(self.inputs, self.inputs + self.outputs)])


class InnovationTracker:
    def __init__(self, start: int = 10000) -> None:
        self.next = start

    def take(self) -> int:
        value = self.next
        self.next += 1
        return value

