"""Deterministic historical replay with independent long and short slot books."""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np


@dataclass
class Position:
    side: int
    entry: float
    leverage: int
    size_fraction: float
    age: int = 0
    mfe: float = 0.0
    mae: float = 0.0


@dataclass
class Result:
    final_equity: float
    realized_pnl: float
    max_drawdown: float
    trades: int
    invalid_actions: int
    long_opens: int
    short_opens: int
    equity_curve: list[float]


def decode(outputs: np.ndarray, ladder: tuple[float, ...]) -> tuple[bool, bool, bool, bool, int, int, float, float]:
    lev = lambda x: int(np.clip(round(2 + (x + 1) * 49), 2, 100))
    size = lambda x: ladder[min(len(ladder) - 1, max(0, int((x + 1) * len(ladder) / 2)))]
    return outputs[0] > .35, outputs[1] > .35, outputs[2] > .45, outputs[3] > .45, lev(outputs[4]), lev(outputs[5]), size(outputs[6]), size(outputs[7])


def run(genome, features: np.ndarray, prices: np.ndarray, *, initial_cash: float = 10000.0, max_slots: int = 3, fee_rate: float = .0005, slippage_rate: float = .0002, ladder=(.01, .02, .05, .1)) -> Result:
    longs: list[Position] = []
    shorts: list[Position] = []
    cash, peak, drawdown, trades, invalid, lo, so = initial_cash, initial_cash, 0.0, 0, 0, 0, 0
    curve: list[float] = []
    previous = np.zeros(4)
    for i in range(21, len(prices)):
        price = float(prices[i])
        unrealized = sum((price / p.entry - 1) * p.side * p.leverage * p.size_fraction * initial_cash for p in longs + shorts)
        state = np.r_[features[i], cash / initial_cash - 1, unrealized / initial_cash, len(longs) / max_slots, len(shorts) / max_slots, previous]
        output = genome.activate(state)
        open_l, open_s, close_l, close_s, lev_l, lev_s, size_l, size_s = decode(output, tuple(ladder))
        for book, close, side in ((longs, close_l, 1), (shorts, close_s, -1)):
            if close:
                if book:
                    p = book.pop(0); exit_price = price * (1 - side * slippage_rate)
                    gross = (exit_price / p.entry - 1) * side * p.leverage * p.size_fraction * initial_cash
                    cash += gross - initial_cash * p.size_fraction * p.leverage * fee_rate
                    trades += 1
                else: invalid += 1
        for book, opening, side, leverage, size in ((longs, open_l, 1, lev_l, size_l), (shorts, open_s, -1, lev_s, size_s)):
            if opening:
                if len(book) < max_slots:
                    book.append(Position(side, price * (1 + side * slippage_rate), leverage, size))
                    cash -= initial_cash * size * leverage * fee_rate
                    if side == 1: lo += 1
                    else: so += 1
                else: invalid += 1
        for p in longs + shorts:
            p.age += 1
            excursion = (price / p.entry - 1) * p.side
            p.mfe, p.mae = max(p.mfe, excursion), min(p.mae, excursion)
        unrealized = sum((price / p.entry - 1) * p.side * p.leverage * p.size_fraction * initial_cash for p in longs + shorts)
        equity = cash + unrealized
        curve.append(equity); peak = max(peak, equity); drawdown = max(drawdown, (peak - equity) / max(peak, 1e-9))
        previous = output[:4]
    # Deterministically liquidate all remaining exposure at the final close. This
    # prevents fitness from rewarding an unclosed, unrealized lottery position.
    price = float(prices[-1])
    for p in longs + shorts:
        exit_price = price * (1 - p.side * slippage_rate)
        cash += (exit_price / p.entry - 1) * p.side * p.leverage * p.size_fraction * initial_cash
        cash -= initial_cash * p.size_fraction * p.leverage * fee_rate
        trades += 1
    final = cash
    if curve:
        curve[-1] = cash
    return Result(final, cash - initial_cash, drawdown, trades, invalid, lo, so, curve)
