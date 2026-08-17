"""Position sizing as a first-class, measurable variable.

Added on evidence, not preference. In the only study of this population
(Wiecki et al. 2016, 888 Quantopian algorithms), volatility, drawdown behaviour
and portfolio construction predicted out-of-sample performance where backtest
Sharpe ratio did not (R^2 < 0.025).

So sizing is not a fixed assumption underneath strategy evaluation. It is a
dimension the harness varies and measures: the same signal at fixed size, at
volatility-targeted size, and under a drawdown constraint is three results.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np

from .strategy import Weights
from .window import MarketWindow


@dataclass(frozen=True)
class PortfolioState:
    """What the sizer is allowed to know about the portfolio."""

    equity: float
    peak_equity: float
    weights: Weights

    @property
    def drawdown(self) -> float:
        """Current drawdown as a positive fraction, e.g. 0.15 for -15%."""
        if self.peak_equity <= 0:
            return 0.0
        return max(0.0, 1.0 - self.equity / self.peak_equity)


@runtime_checkable
class Sizer(Protocol):
    name: str

    def size(
        self, raw: Weights, window: MarketWindow, state: PortfolioState
    ) -> Weights:
        ...


class FixedFraction:
    """Take the strategy's weights as given, scaled by a constant."""

    def __init__(self, scale: float = 1.0) -> None:
        if not 0.0 < scale <= 1.0:
            raise ValueError("scale must be in (0, 1]")
        self.scale = scale
        self.name = f"fixed[{scale:g}]"

    def size(
        self, raw: Weights, window: MarketWindow, state: PortfolioState
    ) -> Weights:
        return {k: v * self.scale for k, v in raw.items()}


class VolatilityTarget:
    """Scale exposure so realised portfolio volatility approaches a target.

    Uses each symbol's own trailing realised volatility. Before enough history
    exists to measure it, exposure is zero — the harness never guesses a number
    it does not have.
    """

    def __init__(
        self,
        target_annual_vol: float = 0.10,
        lookback: int = 60,
        max_leverage: float = 1.0,
    ) -> None:
        self.target = target_annual_vol
        self.lookback = lookback
        self.max_leverage = max_leverage
        self.name = f"voltarget[{target_annual_vol:g},{lookback}d]"

    def size(
        self, raw: Weights, window: MarketWindow, state: PortfolioState
    ) -> Weights:
        out: Weights = {}
        for sym, w in raw.items():
            if w == 0.0:
                out[sym] = 0.0
                continue
            vol = window.realised_vol(sym, self.lookback)
            if not np.isfinite(vol) or vol <= 0.0:
                out[sym] = 0.0
                continue
            scaled = w * (self.target / vol)
            out[sym] = float(np.clip(scaled, -self.max_leverage, self.max_leverage))
        return out


class DrawdownGuard:
    """Wrap another sizer and cut exposure as drawdown deepens.

    Linear de-risking between `start` and `full_stop`: no reduction until the
    portfolio is `start` below its peak, flat by `full_stop`.
    """

    def __init__(self, inner: Sizer, start: float = 0.10, full_stop: float = 0.25):
        if not 0.0 < start < full_stop:
            raise ValueError("require 0 < start < full_stop")
        self.inner = inner
        self.start = start
        self.full_stop = full_stop
        self.name = f"ddguard[{inner.name},{start:g}->{full_stop:g}]"

    def size(
        self, raw: Weights, window: MarketWindow, state: PortfolioState
    ) -> Weights:
        base = self.inner.size(raw, window, state)
        dd = state.drawdown
        if dd <= self.start:
            factor = 1.0
        elif dd >= self.full_stop:
            factor = 0.0
        else:
            factor = 1.0 - (dd - self.start) / (self.full_stop - self.start)
        return {k: v * factor for k, v in base.items()}
