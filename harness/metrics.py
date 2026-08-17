"""Performance metrics.

Every metric is reported against a benchmark, never in isolation. An absolute
return with nothing to compare it to is not a result.

Deliberately absent: win rate. It is trivially gamed by asymmetric payoffs — a
strategy that wins 95% of the time and loses everything on the other 5% has an
excellent win rate and is bankrupt.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

TRADING_DAYS = 252.0


@dataclass(frozen=True)
class Metrics:
    total_return: float
    cagr: float
    ann_vol: float
    sharpe: float
    sortino: float
    max_drawdown: float
    calmar: float
    avg_turnover: float
    cost_drag: float
    n_periods: int

    def as_rows(self) -> list[tuple[str, str]]:
        return [
            ("Total return", f"{self.total_return:+.2%}"),
            ("CAGR", f"{self.cagr:+.2%}"),
            ("Annualised vol", f"{self.ann_vol:.2%}"),
            ("Sharpe", f"{self.sharpe:.3f}"),
            ("Sortino", f"{self.sortino:.3f}"),
            ("Max drawdown", f"{self.max_drawdown:.2%}"),
            ("Calmar", f"{self.calmar:.3f}"),
            ("Avg daily turnover", f"{self.avg_turnover:.4f}"),
            ("Cost drag on terminal wealth", f"{self.cost_drag:.2%}"),
            ("Periods", f"{self.n_periods}"),
        ]


def max_drawdown(equity: np.ndarray) -> float:
    """Deepest peak-to-trough decline, as a positive fraction."""
    if len(equity) == 0:
        return 0.0
    peaks = np.maximum.accumulate(equity)
    return float(np.max(1.0 - equity / peaks))


def sharpe_ratio(returns: np.ndarray, rf_annual: float = 0.0) -> float:
    """Annualised Sharpe. Returns nan rather than a misleading zero."""
    if len(returns) < 2:
        return float("nan")
    excess = returns - rf_annual / TRADING_DAYS
    sd = float(np.std(excess, ddof=1))
    # Not `== 0.0`: a constant series has floating-point noise around 1e-19, and
    # dividing by that produces an enormous Sharpe out of nothing at all.
    if not np.isfinite(sd) or sd < 1e-12:
        return float("nan")
    return float(np.mean(excess) / sd * np.sqrt(TRADING_DAYS))


def sortino_ratio(returns: np.ndarray, rf_annual: float = 0.0) -> float:
    if len(returns) < 2:
        return float("nan")
    excess = returns - rf_annual / TRADING_DAYS
    downside = excess[excess < 0.0]
    if len(downside) < 2:
        return float("nan")
    dd = float(np.std(downside, ddof=1))
    if dd == 0.0:
        return float("nan")
    return float(np.mean(excess) / dd * np.sqrt(TRADING_DAYS))


def compute_metrics(result) -> Metrics:
    """Metrics for a BacktestResult. Typed loosely to avoid a circular import."""
    eq = result.equity
    rets = result.returns
    n = len(rets)

    total = float(eq[-1] / eq[0] - 1.0) if len(eq) > 1 else 0.0
    years = n / TRADING_DAYS if n else 0.0
    if years > 0 and eq[0] > 0 and eq[-1] > 0:
        cagr = float((eq[-1] / eq[0]) ** (1.0 / years) - 1.0)
    else:
        cagr = float("nan")

    vol = float(np.std(rets, ddof=1) * np.sqrt(TRADING_DAYS)) if n > 1 else float("nan")
    mdd = max_drawdown(eq)

    return Metrics(
        total_return=total,
        cagr=cagr,
        ann_vol=vol,
        sharpe=sharpe_ratio(rets),
        sortino=sortino_ratio(rets),
        max_drawdown=mdd,
        calmar=float(cagr / mdd) if mdd > 0 and np.isfinite(cagr) else float("nan"),
        avg_turnover=float(np.mean(result.turnover)) if n else 0.0,
        cost_drag=result.total_cost_drag,
        n_periods=n,
    )


def excess_over(strategy: Metrics, benchmark: Metrics) -> dict[str, float]:
    """The only comparison that matters."""
    return {
        "excess_cagr": strategy.cagr - benchmark.cagr,
        "sharpe_difference": strategy.sharpe - benchmark.sharpe,
        "drawdown_difference": strategy.max_drawdown - benchmark.max_drawdown,
    }
