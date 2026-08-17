"""Report rendering.

No metric appears without its benchmark, its cost drag, and the trial count that
produced it. If those three are missing, the number is decoration.
"""

from __future__ import annotations

from .engine import BacktestResult
from .metrics import Metrics, compute_metrics, excess_over
from .trials import deflation_warning


def _table(rows: list[tuple[str, str]], width: int = 34) -> str:
    return "\n".join(f"  {label:<{width}} {value:>14}" for label, value in rows)


def render_report(
    result: BacktestResult,
    benchmark: BacktestResult,
    dataset: str,
    n_trials: int,
    null_percentile: float | None = None,
) -> str:
    m: Metrics = compute_metrics(result)
    b: Metrics = compute_metrics(benchmark)
    ex = excess_over(m, b)

    lines: list[str] = []
    add = lines.append

    add("=" * 66)
    add(f"  {result.strategy_name}")
    add(f"  sizing: {result.sizer_name}")
    add(f"  data:   {dataset}")
    add("=" * 66)

    warning = deflation_warning(n_trials)
    if warning:
        add("")
        add("  !! " + warning.replace("\n", "\n     "))

    add("")
    add("  STRATEGY")
    add(_table(m.as_rows()))
    add("")
    add(f"  BENCHMARK - {benchmark.strategy_name}")
    add(_table(b.as_rows()))
    add("")
    add("  VERSUS BENCHMARK   <- the only comparison that matters")
    add(
        _table(
            [
                ("Excess CAGR", f"{ex['excess_cagr']:+.2%}"),
                ("Sharpe difference", f"{ex['sharpe_difference']:+.3f}"),
                ("Drawdown difference", f"{ex['drawdown_difference']:+.2%}"),
            ]
        )
    )

    add("")
    add("  HONESTY LAYER")
    rows = [("Trials, this family", str(n_trials))]
    if null_percentile is None:
        rows.append(("Percentile vs random null", "M6 NOT BUILT"))
    else:
        rows.append(("Percentile vs random null", f"{null_percentile:.1%}"))
    rows.append(("Deflated Sharpe", "M8 NOT BUILT"))
    rows.append(("Walk-forward", "M7 NOT BUILT"))
    add(_table(rows))

    add("")
    verdict = (
        "BEATS BENCHMARK on Sharpe"
        if ex["sharpe_difference"] > 0
        else "DOES NOT BEAT BENCHMARK"
    )
    add(f"  VERDICT: {verdict}")
    if null_percentile is None:
        add("           ...and this is NOT yet evidence of an edge.")
        add("           Until M6 exists there is nothing to compare it to but luck.")
    add("=" * 66)

    return "\n".join(lines)
