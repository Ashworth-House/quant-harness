"""MarketWindow — the no-lookahead boundary.

This is the most important object in the harness. A strategy is handed a window
ending at day `t` and can reach nothing beyond it. Every accessor slices to
`t + 1` exclusive, so future data is not merely discouraged, it is not reachable
through the public API.

Python cannot make this airtight against a determined caller, so the structural
guarantee is backed by an empirical one: `harness.testing.assert_no_lookahead`
corrupts all data after `t`, re-runs the strategy, and requires identical output.
Any strategy that peeks fails that test loudly.
"""

from __future__ import annotations

import numpy as np

from .data import Panel


class MarketWindow:
    """A read-only view of a Panel truncated at day `t` (inclusive)."""

    __slots__ = ("_panel", "_t")

    def __init__(self, panel: Panel, t: int) -> None:
        if not 0 <= t < len(panel):
            raise IndexError(f"t={t} outside panel of length {len(panel)}")
        self._panel = panel
        self._t = t

    @property
    def t(self) -> int:
        """Index of the most recent observable day."""
        return self._t

    @property
    def date(self) -> int:
        return int(self._panel.dates[self._t])

    @property
    def symbols(self) -> list[str]:
        return self._panel.symbols

    def __len__(self) -> int:
        """Number of observable days, i.e. t + 1."""
        return self._t + 1

    def opens(self, symbol: str) -> np.ndarray:
        return self._panel.opens[symbol][: self._t + 1]

    def closes(self, symbol: str) -> np.ndarray:
        return self._panel.closes[symbol][: self._t + 1]

    def last_close(self, symbol: str) -> float:
        return float(self._panel.closes[symbol][self._t])

    def log_returns(self, symbol: str) -> np.ndarray:
        """Close-to-close log returns over the observable window."""
        c = self.closes(symbol)
        if len(c) < 2:
            return np.empty(0, dtype=float)
        return np.diff(np.log(c))

    def realised_vol(self, symbol: str, lookback: int) -> float:
        """Annualised realised volatility over the last `lookback` returns.

        Returns nan when there is not enough history — callers must handle it
        rather than silently substituting a number.
        """
        r = self.log_returns(symbol)
        if len(r) < lookback:
            return float("nan")
        return float(np.std(r[-lookback:], ddof=1) * np.sqrt(252.0))
