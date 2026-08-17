"""The strategy contract, and the three baselines that test the harness.

A strategy answers exactly one question: given everything up to today's close,
what do you want to hold from tomorrow's open?

It does not choose its execution price, apply its own costs, size its own
position, or compute its own performance. Those belong to the harness, which is
the only way results from different strategies stay comparable.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np

from .window import MarketWindow

# Desired portfolio weights keyed by symbol. 1.0 means 100% of equity long.
# A missing symbol means a zero weight. The engine rejects gross exposure > 1.0
# because v1 is unlevered by design.
Weights = dict[str, float]


@runtime_checkable
class Strategy(Protocol):
    name: str

    def target_weights(self, window: MarketWindow) -> Weights:
        """Weights to hold from the open of day t+1, decided at the close of t."""
        ...


class BuyAndHold:
    """The correctness check.

    If this does not reproduce the benchmark through the engine, the harness has
    a bug and every number it has ever produced is fiction.
    """

    def __init__(self, symbol: str) -> None:
        self.symbol = symbol
        self.name = f"buy_and_hold[{symbol}]"

    def target_weights(self, window: MarketWindow) -> Weights:
        return {self.symbol: 1.0}


class MovingAverageCrossover:
    """The plausible-sounding strategy that should lose after costs.

    Long when the fast average is above the slow one, flat otherwise. Popular,
    intuitive, and the harness's job is to show it does not beat buy-and-hold.
    """

    def __init__(self, symbol: str, fast: int = 50, slow: int = 200) -> None:
        if fast >= slow:
            raise ValueError("fast window must be shorter than slow window")
        self.symbol = symbol
        self.fast = fast
        self.slow = slow
        self.name = f"ma_crossover[{symbol},{fast}/{slow}]"

    def target_weights(self, window: MarketWindow) -> Weights:
        c = window.closes(self.symbol)
        if len(c) < self.slow:
            return {}
        fast = float(np.mean(c[-self.fast :]))
        slow = float(np.mean(c[-self.slow :]))
        return {self.symbol: 1.0 if fast > slow else 0.0}


class RandomPositions:
    """Coin-flip exposure. The input to the null distribution.

    Seeded, so a single run is reproducible; the point is to run it a thousand
    times with a thousand seeds and look at the spread of the results.
    """

    def __init__(self, symbol: str, seed: int, p_long: float = 0.5) -> None:
        self.symbol = symbol
        self.seed = seed
        self.p_long = p_long
        self.name = f"random[{symbol},seed={seed}]"
        self._rng = np.random.default_rng(seed)

    def target_weights(self, window: MarketWindow) -> Weights:
        return {self.symbol: 1.0 if self._rng.random() < self.p_long else 0.0}

    def reset(self) -> None:
        self._rng = np.random.default_rng(self.seed)
