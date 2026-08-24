"""Risk-aware fitness that rejects inactivity and action spam."""
from __future__ import annotations


def score(result, initial_cash: float = 10000.0) -> float:
    profit = (result.final_equity - initial_cash) / initial_cash
    inactivity = 0.15 if result.trades == 0 else 0.0
    overtrade = max(0, result.trades - 200) * 0.001
    return profit * 100.0 - result.max_drawdown * 40.0 - result.invalid_actions * 0.01 - inactivity - overtrade

