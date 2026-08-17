"""The backtest engine: costs, fills, and the loop.

Two rules are enforced here and nowhere else, because they are the two places
backtests most commonly manufacture returns from nothing:

1. A decision made using data up to the close of day t is filled at the OPEN of
   day t+1. Never at the close that generated the signal.
2. Every rebalance pays commission, half-spread and slippage on the traded
   value. Costs are never netted silently into the return series.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .data import Panel
from .sizing import FixedFraction, PortfolioState, Sizer
from .strategy import Strategy, Weights
from .window import MarketWindow


@dataclass(frozen=True)
class CostModel:
    """Round-trip costs expressed in basis points of traded value.

    Defaults are deliberately pessimistic for a UK retail account. Optimistic
    cost assumptions are the second most common way to fake a backtest.
    """

    commission_bps: float = 5.0
    half_spread_bps: float = 5.0
    slippage_bps: float = 2.0

    @property
    def per_unit_turnover(self) -> float:
        return (self.commission_bps + self.half_spread_bps + self.slippage_bps) / 1e4


@dataclass
class BacktestResult:
    strategy_name: str
    sizer_name: str
    dates: np.ndarray
    equity: np.ndarray
    gross_equity: np.ndarray
    turnover: np.ndarray
    costs_paid: np.ndarray
    exposure: np.ndarray
    meta: dict = field(default_factory=dict)

    @property
    def returns(self) -> np.ndarray:
        if len(self.equity) < 2:
            return np.empty(0, dtype=float)
        return np.diff(self.equity) / self.equity[:-1]

    @property
    def total_cost_drag(self) -> float:
        """Terminal wealth lost to costs, as a fraction of the gross result."""
        if self.gross_equity[-1] == 0:
            return 0.0
        return 1.0 - self.equity[-1] / self.gross_equity[-1]


def run_backtest(
    panel: Panel,
    strategy: Strategy,
    costs: CostModel | None = None,
    sizer: Sizer | None = None,
    initial_equity: float = 1.0,
) -> BacktestResult:
    """Step through the panel one day at a time and record what happened."""
    costs = costs if costs is not None else CostModel()
    sizer = sizer if sizer is not None else FixedFraction()

    n = len(panel)
    if n < 3:
        raise ValueError("need at least 3 days to decide, fill and hold")

    symbols = panel.symbols
    equity = initial_equity
    gross = initial_equity
    peak = initial_equity
    current: Weights = {s: 0.0 for s in symbols}

    dates_out: list[int] = []
    equity_out: list[float] = []
    gross_out: list[float] = []
    turnover_out: list[float] = []
    costs_out: list[float] = []
    exposure_out: list[float] = []

    # Decide at close of t; fill at open of t+1; hold to open of t+2.
    for t in range(n - 2):
        window = MarketWindow(panel, t)
        state = PortfolioState(equity=equity, peak_equity=peak, weights=dict(current))

        raw = strategy.target_weights(window)
        target = sizer.size(raw, window, state)
        target = {s: float(target.get(s, 0.0)) for s in symbols}

        gross_exposure = sum(abs(w) for w in target.values())
        if gross_exposure > 1.0 + 1e-9:
            raise ValueError(
                f"{strategy.name}: gross exposure {gross_exposure:.4f} exceeds 1.0; "
                "v1 is unlevered by design"
            )

        turnover = sum(abs(target[s] - current[s]) for s in symbols)
        cost = turnover * costs.per_unit_turnover

        # Asset returns over the holding period, open[t+1] -> open[t+2].
        period_ret = 0.0
        asset_rets: dict[str, float] = {}
        for s in symbols:
            o1 = panel.opens[s][t + 1]
            o2 = panel.opens[s][t + 2]
            r = (o2 / o1) - 1.0
            asset_rets[s] = r
            period_ret += target[s] * r

        equity = equity * (1.0 - cost) * (1.0 + period_ret)
        gross = gross * (1.0 + period_ret)
        peak = max(peak, equity)

        # Let held weights drift with the period's returns.
        denom = 1.0 + period_ret
        if abs(denom) < 1e-12:
            current = {s: 0.0 for s in symbols}
        else:
            current = {s: target[s] * (1.0 + asset_rets[s]) / denom for s in symbols}

        dates_out.append(int(panel.dates[t + 2]))
        equity_out.append(equity)
        gross_out.append(gross)
        turnover_out.append(turnover)
        costs_out.append(cost)
        exposure_out.append(gross_exposure)

    return BacktestResult(
        strategy_name=strategy.name,
        sizer_name=sizer.name,
        dates=np.array(dates_out, dtype=np.int64),
        equity=np.array(equity_out, dtype=float),
        gross_equity=np.array(gross_out, dtype=float),
        turnover=np.array(turnover_out, dtype=float),
        costs_paid=np.array(costs_out, dtype=float),
        exposure=np.array(exposure_out, dtype=float),
        meta={"n_days": n, "cost_model": costs},
    )
