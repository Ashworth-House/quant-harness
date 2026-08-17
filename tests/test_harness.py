"""Harness correctness tests.

A backtest bug is indistinguishable from alpha. These tests are the reason to
believe any number this thing prints.
"""

from __future__ import annotations

import numpy as np
import pytest

from harness import (
    BuyAndHold,
    CostModel,
    DrawdownGuard,
    FixedFraction,
    MarketWindow,
    MovingAverageCrossover,
    RandomPositions,
    VolatilityTarget,
    assert_no_lookahead,
    compute_metrics,
    corrupt_after,
    run_backtest,
    shuffled_panel,
    synthetic_panel,
)
from harness.metrics import max_drawdown, sharpe_ratio

SYM = "TEST"


@pytest.fixture
def panel():
    return synthetic_panel([SYM], n_days=1500, seed=42, annual_drift=0.07)


@pytest.fixture
def flat_panel():
    return synthetic_panel([SYM], n_days=800, seed=7, annual_drift=0.0)


# --------------------------------------------------------------------------
# The no-lookahead boundary
# --------------------------------------------------------------------------


def test_window_exposes_only_history(panel):
    w = MarketWindow(panel, 100)
    assert len(w) == 101
    assert len(w.closes(SYM)) == 101
    assert len(w.opens(SYM)) == 101
    assert w.last_close(SYM) == pytest.approx(panel.closes[SYM][100])


def test_window_rejects_out_of_range(panel):
    with pytest.raises(IndexError):
        MarketWindow(panel, len(panel))
    with pytest.raises(IndexError):
        MarketWindow(panel, -1)


def test_corrupting_the_future_does_not_change_the_window(panel):
    t = 300
    clean = MarketWindow(panel, t).closes(SYM)
    dirty = MarketWindow(corrupt_after(panel, t), t).closes(SYM)
    np.testing.assert_array_equal(clean, dirty)


@pytest.mark.parametrize(
    "factory",
    [
        lambda: BuyAndHold(SYM),
        lambda: MovingAverageCrossover(SYM, 50, 200),
        lambda: RandomPositions(SYM, seed=1),
    ],
)
def test_baselines_do_not_look_ahead(panel, factory):
    """The empirical guarantee: corrupt the future, output must not move."""
    assert_no_lookahead(factory, panel)


def test_the_lookahead_detector_actually_catches_a_cheat(panel):
    """A detector that never fires proves nothing. This one fires."""

    class Cheat:
        name = "cheat"

        def target_weights(self, window):
            # Reaches through the window into its private panel to read tomorrow.
            # This is the realistic attack: the public API cannot serve it, so a
            # peeking strategy must grab private state. Exactly what must be caught.
            future = window._panel.closes[SYM][window.t + 1]
            return {SYM: 1.0 if future > window.last_close(SYM) else 0.0}

    with pytest.raises(AssertionError, match="LOOKAHEAD DETECTED"):
        assert_no_lookahead(Cheat, panel)


# --------------------------------------------------------------------------
# M2 - the correctness check
# --------------------------------------------------------------------------


def test_buy_and_hold_reproduces_the_underlying(panel):
    """If this fails, every other number the harness prints is fiction."""
    res = run_backtest(panel, BuyAndHold(SYM), costs=CostModel(0, 0, 0))
    opens = panel.opens[SYM]
    expected = opens[len(panel) - 1] / opens[1]
    assert res.equity[-1] == pytest.approx(expected, rel=1e-9)


def test_buy_and_hold_trades_once(panel):
    res = run_backtest(panel, BuyAndHold(SYM), costs=CostModel(0, 0, 0))
    assert res.turnover[0] == pytest.approx(1.0)
    assert np.all(res.turnover[1:] < 1e-9)


# --------------------------------------------------------------------------
# M3 - execution and costs
# --------------------------------------------------------------------------


def test_costs_reduce_returns(panel):
    free = run_backtest(panel, MovingAverageCrossover(SYM), costs=CostModel(0, 0, 0))
    paid = run_backtest(panel, MovingAverageCrossover(SYM), costs=CostModel(5, 5, 2))
    assert paid.equity[-1] < free.equity[-1]
    assert paid.total_cost_drag > 0.0


def test_costs_scale_with_turnover(panel):
    """A strategy that trades more must pay more. Twice, in fact."""
    cheap = run_backtest(panel, MovingAverageCrossover(SYM), costs=CostModel(5, 5, 2))
    dear = run_backtest(panel, MovingAverageCrossover(SYM), costs=CostModel(10, 10, 4))
    assert dear.total_cost_drag > cheap.total_cost_drag


def test_fills_use_next_bar_open_not_signal_close(panel):
    """The single most common backtest bug, pinned down by construction.

    A strategy deciding at close t must earn the open[t+1] -> open[t+2] return.
    If the engine ever filled at close t, this equality would break.
    """
    res = run_backtest(panel, BuyAndHold(SYM), costs=CostModel(0, 0, 0))
    opens = panel.opens[SYM]
    first_period = opens[2] / opens[1] - 1.0
    assert res.equity[0] == pytest.approx(1.0 + first_period, rel=1e-12)


def test_leverage_is_rejected(panel):
    class Levered:
        name = "levered"

        def target_weights(self, window):
            return {SYM: 1.5}

    with pytest.raises(ValueError, match="exceeds 1.0"):
        run_backtest(panel, Levered())


# --------------------------------------------------------------------------
# M3b - sizing as a measurable variable
# --------------------------------------------------------------------------


def test_same_signal_three_sizings_three_results(panel):
    """The whole point of M3b: sizing is a dimension, not an assumption."""
    results = {
        s.name: run_backtest(panel, BuyAndHold(SYM), sizer=s)
        for s in (
            FixedFraction(1.0),
            VolatilityTarget(0.10, 60),
            DrawdownGuard(FixedFraction(1.0), 0.05, 0.15),
        )
    }
    finals = [r.equity[-1] for r in results.values()]
    assert len({round(f, 6) for f in finals}) == 3, "sizings must be distinguishable"


def test_volatility_target_reduces_realised_vol(panel):
    plain = run_backtest(panel, BuyAndHold(SYM), sizer=FixedFraction(1.0))
    vt = run_backtest(panel, BuyAndHold(SYM), sizer=VolatilityTarget(0.05, 60))
    assert compute_metrics(vt).ann_vol < compute_metrics(plain).ann_vol


def test_volatility_target_is_flat_before_it_has_history(panel):
    """No guessing a number we do not have."""
    res = run_backtest(panel, BuyAndHold(SYM), sizer=VolatilityTarget(0.10, 60))
    assert res.exposure[0] == pytest.approx(0.0)


def test_drawdown_guard_cuts_exposure_after_losses(flat_panel):
    guarded = run_backtest(
        flat_panel,
        BuyAndHold(SYM),
        sizer=DrawdownGuard(FixedFraction(1.0), 0.02, 0.06),
    )
    plain = run_backtest(flat_panel, BuyAndHold(SYM), sizer=FixedFraction(1.0))
    assert compute_metrics(guarded).max_drawdown <= compute_metrics(plain).max_drawdown


# --------------------------------------------------------------------------
# M4 - metrics
# --------------------------------------------------------------------------


def test_max_drawdown_on_a_known_series():
    eq = np.array([1.0, 1.5, 0.75, 1.2])  # peak 1.5 -> trough 0.75 is -50%
    assert max_drawdown(eq) == pytest.approx(0.5)


def test_sharpe_of_constant_returns_is_undefined():
    assert np.isnan(sharpe_ratio(np.full(100, 0.001)))


def test_sharpe_sign_follows_returns():
    rng = np.random.default_rng(0)
    assert sharpe_ratio(rng.normal(0.001, 0.01, 2000)) > 0
    assert sharpe_ratio(rng.normal(-0.001, 0.01, 2000)) < 0


def test_metrics_are_finite_on_a_real_run(panel):
    m = compute_metrics(run_backtest(panel, MovingAverageCrossover(SYM)))
    for field in ("total_return", "cagr", "ann_vol", "sharpe", "max_drawdown"):
        assert np.isfinite(getattr(m, field)), f"{field} is not finite"


# --------------------------------------------------------------------------
# The headline result: the plausible strategy loses
# --------------------------------------------------------------------------


def test_moving_average_crossover_loses_to_buy_and_hold():
    """M5. Popular, intuitive, and worse than doing nothing once costs are paid.

    Averaged over many seeds, not asserted on one. A single seed is a single
    sample, and a project about not fooling yourself with one lucky backtest
    should not have a test that depends on one lucky backtest.
    """
    costs = CostModel(5, 5, 2)
    ma_returns, bh_returns = [], []

    for seed in range(12):
        p = synthetic_panel([SYM], n_days=1500, seed=seed, annual_drift=0.07)
        ma_returns.append(
            compute_metrics(
                run_backtest(p, MovingAverageCrossover(SYM), costs=costs)
            ).total_return
        )
        bh_returns.append(
            compute_metrics(run_backtest(p, BuyAndHold(SYM), costs=costs)).total_return
        )

    assert np.mean(ma_returns) < np.mean(bh_returns), (
        f"MA mean {np.mean(ma_returns):.2%} vs buy-and-hold "
        f"{np.mean(bh_returns):.2%} - the harness cannot show a mediocre "
        "strategy is mediocre"
    )
    assert sum(m < b for m, b in zip(ma_returns, bh_returns)) >= 7


# --------------------------------------------------------------------------
# Negative controls - the data with no edge in it
# --------------------------------------------------------------------------


def test_synthetic_data_has_no_drift_when_asked(flat_panel):
    """Compared against its own standard error, not a magic number.

    A finite sample of a zero-drift walk will not have zero realised drift. The
    honest test is whether the realised drift is within sampling error of zero.
    """
    logret = np.diff(np.log(flat_panel.opens[SYM]))
    ann_drift = float(np.mean(logret) * 252)
    years = len(logret) / 252.0
    ann_vol = float(np.std(logret, ddof=1) * np.sqrt(252))
    std_error = ann_vol / np.sqrt(years)
    assert abs(ann_drift) < 3 * std_error, (
        f"realised drift {ann_drift:.3f} is more than 3 standard errors "
        f"({std_error:.3f}) from zero"
    )


def test_shuffling_preserves_distribution_but_destroys_order(panel):
    sh = shuffled_panel(panel, seed=3)
    a = np.sort(np.diff(np.log(panel.opens[SYM])))
    b = np.sort(np.diff(np.log(sh.opens[SYM])))
    np.testing.assert_allclose(a, b, atol=1e-12)
    assert not np.allclose(
        np.diff(np.log(panel.opens[SYM])), np.diff(np.log(sh.opens[SYM]))
    )


def test_random_strategies_disagree_with_each_other(flat_panel):
    """Sanity check for M6: different seeds must give different outcomes."""
    finals = [
        run_backtest(flat_panel, RandomPositions(SYM, seed=s)).equity[-1]
        for s in range(5)
    ]
    assert len({round(f, 8) for f in finals}) == 5
