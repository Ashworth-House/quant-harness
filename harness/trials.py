"""Trial counting.

With five years of daily data, roughly 45 trials is enough to find a strategy
with an in-sample Sharpe of 1 whose expected out-of-sample Sharpe is zero
(Bailey, Borwein, Lopez de Prado & Zhu, Notices of the AMS, 2014).

So the number of configurations tested is not bookkeeping — it is an input to
whether any result means anything. Without the count, the deflated Sharpe ratio
cannot be computed and a backtest is uninterpretable.

The count is recorded by the harness, not by hand, because a count maintained
by hand is a count that flatters the person maintaining it.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

TRIALS_FILE = "TRIALS.jsonl"


def record_trial(
    strategy_name: str,
    sizer_name: str,
    dataset: str,
    sharpe: float,
    root: str = ".",
) -> int:
    """Append one trial and return the new total for this strategy family.

    Family is the strategy name up to the first '[' — so ma_crossover[SPY,50/200]
    and ma_crossover[SPY,20/100] count as trials of the same idea, which is
    exactly the point. Tuning parameters IS searching.
    """
    path = Path(root) / TRIALS_FILE
    family = strategy_name.split("[")[0]

    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "family": family,
        "strategy": strategy_name,
        "sizer": sizer_name,
        "dataset": dataset,
        "sharpe": None if sharpe != sharpe else round(float(sharpe), 6),
    }

    with path.open("a") as fh:
        fh.write(json.dumps(entry) + "\n")

    return count_trials(family, root=root)


def count_trials(family: str | None = None, root: str = ".") -> int:
    """Trials recorded for a family, or in total when family is None."""
    path = Path(root) / TRIALS_FILE
    if not path.exists():
        return 0
    n = 0
    with path.open() as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            if family is None or json.loads(line).get("family") == family:
                n += 1
    return n


def deflation_warning(n_trials: int) -> str | None:
    """A blunt reminder, printed on every report.

    Not a substitute for the deflated Sharpe ratio (M8) — a stand-in until it
    exists, so that nobody reads a raw Sharpe in the meantime without flinching.
    """
    if n_trials >= 45:
        return (
            f"{n_trials} trials recorded for this family. At this many trials an "
            "in-sample Sharpe near 1.0 is consistent with ZERO true edge "
            "(Bailey et al. 2014). Raw Sharpe below is not interpretable without "
            "deflation - see M8."
        )
    if n_trials >= 10:
        return (
            f"{n_trials} trials recorded for this family. Selection effects are "
            "already material. Treat the raw Sharpe as an upper bound."
        )
    return None
