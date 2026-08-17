"""M6 — the random-strategy null distribution.

=============================================================================
THIS MODULE IS DELIBERATELY UNIMPLEMENTED. IT IS THE CO-FOUNDER'S TO BUILD.
=============================================================================

Everything else in this harness is scaffolding. This is the part that makes the
whole thing worth something, and it is the piece to be defended in an interview,
so it should be built by the person who will be defending it.

WHAT IT DOES
------------
Run a coin-flip strategy `n_runs` times with `n_runs` different seeds. Collect
the Sharpe ratio of each run. That distribution is what *luck alone* looks like
on this data, with these costs, over this period.

Then any real strategy's Sharpe can be placed on it. If it sits inside the
random cloud, it has no demonstrated edge — however good its equity curve looks
in isolation.

WHY IT MATTERS
--------------
A Sharpe of 1.2 means nothing until you know that random trading on the same
data produces a Sharpe above 1.2 roughly one time in twenty. This module is
what turns a number into a claim.

DONE WHEN
---------
`tests/test_m6_null_distribution.py` passes with its module-level skip removed.
Read that file first — it is the specification.

THE STATISTICS TO UNDERSTAND BEFORE WRITING IT
----------------------------------------------
Sampling distributions, the bootstrap, and permutation tests. This function IS a
permutation test; building it without understanding it produces a chart rather
than an argument. See Planning Documents/learning-path.md topic 1.

A HINT ON WHAT WILL SURPRISE YOU
--------------------------------
The null distribution will probably not be centred on a Sharpe of zero, and
working out why is the most instructive part of the exercise. Consider what a
long-only coin-flip strategy is actually exposed to when the underlying asset
has a drift, and what that implies about the right null for a long-only
strategy versus a long-short one.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .data import Panel
from .engine import CostModel
from .sizing import Sizer


@dataclass(frozen=True)
class NullDistribution:
    """The distribution of a metric under random trading."""

    sharpes: np.ndarray
    n_runs: int
    symbol: str

    def percentile_of(self, sharpe: float) -> float:
        """Fraction of random runs this value beats, in [0, 1].

        0.97 means the strategy beat 97% of coin flips. Below roughly 0.95,
        there is no evidence of an edge.
        """
        raise NotImplementedError("M6 — see module docstring")

    def summary(self) -> dict[str, float]:
        """At minimum: mean, std, and the 5th/50th/95th percentiles."""
        raise NotImplementedError("M6 — see module docstring")


def build_null_distribution(
    panel: Panel,
    symbol: str,
    n_runs: int = 1000,
    costs: CostModel | None = None,
    sizer: Sizer | None = None,
    base_seed: int = 0,
) -> NullDistribution:
    """Run RandomPositions `n_runs` times and collect the Sharpe of each.

    Use a different seed per run, derived deterministically from `base_seed`, so
    the whole distribution is reproducible.
    """
    raise NotImplementedError("M6 — see module docstring")


def plot_null_distribution(
    null: NullDistribution,
    mark: dict[str, float] | None = None,
    path: str = "null_distribution.png",
) -> str:
    """Histogram of the null, with optional vertical markers for real strategies.

    `mark` maps a strategy name to its Sharpe. Returns the path written.

    This is the single most valuable image the project will produce. Make it
    legible: label the axes, state n_runs, and show where each marked strategy
    falls as a percentile, not just a line.
    """
    raise NotImplementedError("M6 — see module docstring")
