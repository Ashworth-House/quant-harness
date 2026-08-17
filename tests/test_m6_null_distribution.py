"""M6 — the specification for the null distribution. THIS FILE IS THE BRIEF.

=============================================================================
  These tests are skipped. Delete the `pytest.skip` line below and make them
  pass. That is the definition of "M6 done".
=============================================================================

Read `harness/nulldist.py` first. Then read Planning Documents/learning-path.md
topic 1 — resampling and permutation tests — because this module IS a
permutation test and building it without understanding it produces a chart
rather than an argument.

Nothing here is a trick. If the tests seem to be asking for something odd,
the odd thing is probably the real statistical point.
"""

from __future__ import annotations

import numpy as np
import pytest

from harness import CostModel, MovingAverageCrossover, compute_metrics, run_backtest
from harness.data import synthetic_panel
from harness.nulldist import build_null_distribution

pytest.skip(
    "M6 is not implemented yet — this file is its specification. "
    "Delete this skip when you start.",
    allow_module_level=True,
)

SYM = "TEST"
COSTS = CostModel(5, 5, 2)


@pytest.fixture(scope="module")
def panel():
    return synthetic_panel([SYM], n_days=1200, seed=11, annual_drift=0.05)


@pytest.fixture(scope="module")
def null(panel):
    # 200 runs keeps the test suite fast. Use 1000+ for real results.
    return build_null_distribution(panel, SYM, n_runs=200, costs=COSTS, base_seed=0)


def test_it_runs_the_number_of_trials_asked_for(null):
    assert null.n_runs == 200
    assert len(null.sharpes) == 200


def test_the_distribution_has_spread(null):
    """If every coin flip gave the same answer, something is wired wrong."""
    assert float(np.std(null.sharpes)) > 0.05


def test_it_is_reproducible(panel):
    a = build_null_distribution(panel, SYM, n_runs=50, costs=COSTS, base_seed=7)
    b = build_null_distribution(panel, SYM, n_runs=50, costs=COSTS, base_seed=7)
    np.testing.assert_allclose(a.sharpes, b.sharpes)


def test_different_base_seeds_give_different_distributions(panel):
    a = build_null_distribution(panel, SYM, n_runs=50, costs=COSTS, base_seed=1)
    b = build_null_distribution(panel, SYM, n_runs=50, costs=COSTS, base_seed=2)
    assert not np.allclose(a.sharpes, b.sharpes)


def test_percentile_is_bounded_and_monotonic(null):
    lo = null.percentile_of(float(np.min(null.sharpes)) - 1.0)
    hi = null.percentile_of(float(np.max(null.sharpes)) + 1.0)
    mid = null.percentile_of(float(np.median(null.sharpes)))
    assert lo == pytest.approx(0.0, abs=0.01)
    assert hi == pytest.approx(1.0, abs=0.01)
    assert 0.4 < mid < 0.6


def test_summary_reports_what_a_reader_needs(null):
    s = null.summary()
    for key in ("mean", "std", "p05", "p50", "p95"):
        assert key in s, f"summary must report {key}"
    assert s["p05"] < s["p50"] < s["p95"]


def test_the_whole_point__a_mediocre_strategy_does_not_stand_out(panel, null):
    """The result that makes the project worth doing.

    Run the moving-average crossover, take its Sharpe, and place it on the null.
    It should sit inside the random cloud — that is, it should NOT be above the
    95th percentile. If a strategy this ordinary cleared the bar, the bar would
    be measuring nothing.
    """
    res = run_backtest(panel, MovingAverageCrossover(SYM), costs=COSTS)
    sharpe = compute_metrics(res).sharpe
    assert null.percentile_of(sharpe) < 0.95


def test_plot_writes_a_file(null, tmp_path):
    from harness.nulldist import plot_null_distribution

    out = plot_null_distribution(
        null, mark={"ma_crossover": 0.4}, path=str(tmp_path / "null.png")
    )
    assert (tmp_path / "null.png").exists()
    assert out.endswith("null.png")
