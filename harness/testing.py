"""Empirical guarantees the type system cannot give us.

MarketWindow makes future data unreachable through its public API, but Python
cannot make that airtight against a caller who reaches for private attributes.
So we test it instead of asserting it: corrupt everything after day t, re-run,
and require identical output.

A strategy that peeks fails loudly. That is a much stronger guarantee than a
convention nobody enforces.
"""

from __future__ import annotations

import numpy as np

from .data import Panel
from .strategy import Strategy, Weights
from .window import MarketWindow


def corrupt_after(panel: Panel, t: int, seed: int = 0) -> Panel:
    """Replace all data strictly after day t with unrelated noise."""
    rng = np.random.default_rng(seed)
    opens: dict[str, np.ndarray] = {}
    closes: dict[str, np.ndarray] = {}

    for sym in panel.symbols:
        o = panel.opens[sym].copy()
        c = panel.closes[sym].copy()
        n_after = len(o) - (t + 1)
        if n_after > 0:
            o[t + 1 :] = rng.uniform(1.0, 1000.0, size=n_after)
            c[t + 1 :] = rng.uniform(1.0, 1000.0, size=n_after)
        opens[sym] = o
        closes[sym] = c

    return Panel(dates=panel.dates.copy(), opens=opens, closes=closes)


def assert_no_lookahead(
    strategy_factory,
    panel: Panel,
    check_days: list[int] | None = None,
) -> None:
    """Verify a strategy's output at day t depends only on data up to day t.

    `strategy_factory` must be a zero-argument callable returning a *fresh*
    strategy, so that stateful strategies start from the same place in both runs.

    Raises AssertionError naming the first day where the outputs diverge.
    """
    n = len(panel)
    if check_days is None:
        check_days = [n // 4, n // 2, (3 * n) // 4, n - 3]
    check_days = [t for t in check_days if 0 <= t < n - 2]

    for t in check_days:
        clean = _weights_at(strategy_factory(), panel, t)
        dirty = _weights_at(strategy_factory(), corrupt_after(panel, t), t)
        if not _weights_equal(clean, dirty):
            raise AssertionError(
                f"LOOKAHEAD DETECTED at t={t}: output changed when future data was "
                f"corrupted.\n  with real future:      {clean}\n"
                f"  with corrupted future: {dirty}"
            )


def _weights_at(strategy: Strategy, panel: Panel, t: int) -> Weights:
    """Replay the strategy up to t so stateful strategies see the same history."""
    out: Weights = {}
    for i in range(t + 1):
        out = strategy.target_weights(MarketWindow(panel, i))
    return out


def _weights_equal(a: Weights, b: Weights, tol: float = 1e-12) -> bool:
    keys = set(a) | set(b)
    return all(abs(a.get(k, 0.0) - b.get(k, 0.0)) <= tol for k in keys)
