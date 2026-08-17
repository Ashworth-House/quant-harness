"""A trustworthy scoreboard for trading strategies.

The goal of v1 is not a profitable strategy. It is an evaluation harness that can
prove a bad strategy is bad — because in the only study of this population
(Wiecki et al., 888 Quantopian algorithms), backtest Sharpe predicted live
performance with R^2 < 0.025. Without honest measurement, a backtest is a story.
"""

from .data import Panel, panel_from_csv, shuffled_panel, synthetic_panel
from .engine import BacktestResult, CostModel, run_backtest
from .metrics import Metrics, compute_metrics, excess_over
from .report import render_report
from .sizing import DrawdownGuard, FixedFraction, PortfolioState, VolatilityTarget
from .strategy import BuyAndHold, MovingAverageCrossover, RandomPositions, Strategy
from .testing import assert_no_lookahead, corrupt_after
from .trials import count_trials, record_trial
from .window import MarketWindow

__all__ = [
    "Panel",
    "synthetic_panel",
    "shuffled_panel",
    "panel_from_csv",
    "MarketWindow",
    "Strategy",
    "BuyAndHold",
    "MovingAverageCrossover",
    "RandomPositions",
    "FixedFraction",
    "VolatilityTarget",
    "DrawdownGuard",
    "PortfolioState",
    "CostModel",
    "BacktestResult",
    "run_backtest",
    "Metrics",
    "compute_metrics",
    "excess_over",
    "render_report",
    "assert_no_lookahead",
    "corrupt_after",
    "record_trial",
    "count_trials",
]
