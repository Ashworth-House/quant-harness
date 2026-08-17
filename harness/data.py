"""Market data: a minimal panel, a synthetic generator, and negative controls.

The harness is developed and tested against *synthetic* data on purpose. It costs
nothing, carries no licence restrictions, has no survivorship bias, and — most
usefully — synthetic series with no exploitable structure are exactly what you need
to measure your own false-discovery rate. Real data slots in behind the same Panel.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Panel:
    """Aligned OHLCV for one or more symbols.

    Every array has the same length and shares one date index. Nothing in the
    harness reads a Panel directly except the engine, which wraps it in a
    MarketWindow before any strategy sees it.
    """

    dates: np.ndarray
    opens: dict[str, np.ndarray]
    closes: dict[str, np.ndarray]

    def __post_init__(self) -> None:
        n = len(self.dates)
        for sym in self.symbols:
            if len(self.opens[sym]) != n or len(self.closes[sym]) != n:
                raise ValueError(f"{sym}: arrays must match the date index length {n}")
            if not np.all(np.isfinite(self.opens[sym])):
                raise ValueError(f"{sym}: non-finite values in opens")
            if not np.all(np.isfinite(self.closes[sym])):
                raise ValueError(f"{sym}: non-finite values in closes")

    @property
    def symbols(self) -> list[str]:
        return sorted(self.opens)

    def __len__(self) -> int:
        return len(self.dates)


def synthetic_panel(
    symbols: list[str],
    n_days: int,
    seed: int,
    annual_drift: float = 0.0,
    annual_vol: float = 0.20,
    start_price: float = 100.0,
) -> Panel:
    """A geometric random walk. No predictable structure, by construction.

    `annual_drift=0.0` is the honest default for a negative control: any strategy
    that appears to make risk-adjusted money here is measuring your own search
    process, not the market.
    """
    rng = np.random.default_rng(seed)
    # Each DAY advances by dt. A day is two half-steps (open->close, close->next
    # open), so each half-step carries dt/2. Getting this wrong doubles the
    # effective time step and inflates realised volatility by sqrt(2).
    half_dt = (1.0 / 252.0) / 2.0
    mu = annual_drift
    sigma = annual_vol

    dates = np.arange(n_days, dtype=np.int64)
    opens: dict[str, np.ndarray] = {}
    closes: dict[str, np.ndarray] = {}

    for sym in symbols:
        shocks = rng.normal(
            loc=(mu - 0.5 * sigma**2) * half_dt,
            scale=sigma * np.sqrt(half_dt),
            size=n_days * 2,
        )
        path = start_price * np.exp(np.cumsum(shocks))
        # Interleave so each day has an open and a close drawn from one path.
        opens[sym] = path[0::2].copy()
        closes[sym] = path[1::2].copy()

    return Panel(dates=dates, opens=opens, closes=closes)


def shuffled_panel(panel: Panel, seed: int) -> Panel:
    """Destroy time-ordering while preserving the return distribution.

    The second negative control. Real marginal distributions, real fat tails, no
    temporal structure whatsoever — so any time-series edge found here is spurious
    by construction.
    """
    rng = np.random.default_rng(seed)
    opens: dict[str, np.ndarray] = {}
    closes: dict[str, np.ndarray] = {}

    for sym in panel.symbols:
        o = panel.opens[sym]
        rets = np.diff(np.log(o))
        rng.shuffle(rets)
        rebuilt = o[0] * np.exp(np.concatenate([[0.0], np.cumsum(rets)]))
        opens[sym] = rebuilt
        # Preserve each day's open->close relationship, reordered with the path.
        intraday = np.log(panel.closes[sym] / panel.opens[sym])
        rng.shuffle(intraday)
        closes[sym] = rebuilt * np.exp(intraday)

    return Panel(dates=panel.dates.copy(), opens=opens, closes=closes)


def panel_from_csv(path: str, symbol: str) -> Panel:
    """Load a single symbol from CSV with columns: date, open, close.

    Deliberately minimal. When a licensed vendor is chosen, write an adapter that
    produces a Panel and nothing else in the harness needs to change.

    NOTE: no licensed market data is ever committed to this repository.
    """
    import csv

    dates: list[int] = []
    opens: list[float] = []
    closes: list[float] = []

    with open(path, newline="") as fh:
        for i, row in enumerate(csv.DictReader(fh)):
            dates.append(i)
            opens.append(float(row["open"]))
            closes.append(float(row["close"]))

    return Panel(
        dates=np.array(dates, dtype=np.int64),
        opens={symbol: np.array(opens, dtype=float)},
        closes={symbol: np.array(closes, dtype=float)},
    )
