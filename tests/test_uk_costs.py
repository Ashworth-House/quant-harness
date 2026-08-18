"""UK-specific costs and investability.

Both of these fail SILENTLY if unmodelled, and both produce a plausible,
entirely fictitious equity curve. That is why they are tests rather than notes.
"""

from __future__ import annotations

import numpy as np
import pytest

from harness import BuyAndHold, CostModel, compute_metrics, run_backtest, synthetic_panel
from harness.engine import UK_ETF_IBKR, UK_SHARES_IBKR
from harness.instruments import (
    NotInvestable,
    assert_investable,
    uk_share,
    ucits_etf,
    us_domiciled_etf,
)

SYM = "TEST"


# ---------------------------------------------------------------- investability


def test_us_domiciled_etfs_are_refused_by_name():
    with pytest.raises(NotInvestable, match="SPY"):
        assert_investable(["SPY", "AAPL"])


def test_the_refusal_explains_itself_and_offers_the_substitute():
    with pytest.raises(NotInvestable) as e:
        assert_investable(["VOO"])
    msg = str(e.value)
    assert "Overseas Funds Regime" in msg
    assert "CSPX" in msg, "must name an investable substitute, not just refuse"


def test_investable_universes_pass():
    assert_investable(["CSPX", "VUSA", "EQQQ", "AAPL"])


def test_unknown_symbols_pass():
    """We cannot know every instrument. Only the known traps fail."""
    assert_investable(["SOME_OBSCURE_TICKER"])


# ---------------------------------------------------------------- stamp duty


def test_stamp_duty_applies_to_buys_only():
    """The asymmetry is the whole point. A symmetric round-trip estimate is wrong."""
    costs = CostModel(0, 0, 0)
    meta = uk_share(SYM)
    buy = costs.cost_for_trade(10_000.0, is_buy=True, meta=meta)
    sell = costs.cost_for_trade(10_000.0, is_buy=False, meta=meta)
    assert buy == pytest.approx(50.0)   # 0.5% of 10,000
    assert sell == pytest.approx(0.0)


def test_etfs_are_exempt_from_stamp_duty():
    costs = CostModel(0, 0, 0)
    assert costs.cost_for_trade(10_000.0, True, ucits_etf("CSPX")) == pytest.approx(0.0)
    assert costs.cost_for_trade(10_000.0, True, uk_share("BP")) == pytest.approx(50.0)


def test_stamp_duty_is_per_instrument_not_per_exchange():
    """Both LSE-listed. Only one is UK-incorporated. Venue cannot tell you this."""
    costs = CostModel(0, 0, 0)
    uk = costs.cost_for_trade(5_000.0, True, uk_share("LSE_UK_CO"))
    etf = costs.cost_for_trade(5_000.0, True, ucits_etf("LSE_ETF"))
    assert uk > 0 and etf == 0


def test_stamp_duty_drag_on_a_rebalancing_uk_share_strategy():
    """Quantify it, because the number is the argument.

    A strategy holding UK-incorporated shares and turning the book over pays a
    tax no amount of signal quality recovers.
    """
    panel = synthetic_panel([SYM], n_days=1000, seed=5, annual_drift=0.07)

    class Churner:
        name = "churner"

        def __init__(self):
            self.on = False

        def target_weights(self, w):
            self.on = not self.on
            return {SYM: 1.0 if self.on else 0.0}

    free = run_backtest(panel, Churner(), costs=CostModel(0, 0, 0))
    taxed = run_backtest(
        panel, Churner(), costs=CostModel(0, 0, 0), instruments={SYM: uk_share(SYM)}
    )
    assert taxed.equity[-1] < free.equity[-1]
    annual_drag = 1.0 - (taxed.equity[-1] / free.equity[-1]) ** (252 / len(taxed.equity))
    assert annual_drag > 0.10, f"expected heavy drag, got {annual_drag:.2%}"


# ---------------------------------------------------------------- order minimums


def test_commission_minimum_dominates_at_small_position_size():
    """At retail size the binding cost is the order minimum, not the rate."""
    costs = CostModel(commission_bps=5.0, half_spread_bps=0, slippage_bps=0,
                      commission_min=1.00)
    meta = ucits_etf("CSPX")

    small = costs.cost_for_trade(250.0, True, meta)     # 5bp would be 0.125
    large = costs.cost_for_trade(10_000.0, True, meta)  # 5bp is 5.00

    assert small == pytest.approx(1.00), "minimum must bind on a small order"
    assert large == pytest.approx(5.00), "rate must bind on a large order"
    assert small / 250.0 > large / 10_000.0, "small orders cost more in bps"


def test_cost_in_bps_falls_as_position_size_rises():
    """The diversification-versus-cost trade at small capital, in one assertion."""
    costs = UK_ETF_IBKR
    meta = ucits_etf("CSPX")
    bps = [
        1e4 * costs.cost_for_trade(v, True, meta) / v
        for v in (250.0, 500.0, 750.0, 1500.0, 2600.0)
    ]
    assert bps == sorted(bps, reverse=True), "cost in bps must fall with order size"
    assert bps[0] > 2 * bps[-1], "the effect must be large, not marginal"


def test_ptm_levy_applies_only_above_its_threshold():
    costs = UK_SHARES_IBKR
    meta = ucits_etf("X")
    below = costs.cost_for_trade(9_000.0, True, meta)
    above = costs.cost_for_trade(11_000.0, True, meta)
    assert above - below > 1.0


# ---------------------------------------------------------------- regression


def test_default_cost_model_is_unchanged():
    """The upgrade must not silently move every previously recorded result."""
    panel = synthetic_panel([SYM], n_days=600, seed=9, annual_drift=0.05)
    res = run_backtest(panel, BuyAndHold(SYM), costs=CostModel(5, 5, 2))
    expected_first = 12.0 / 1e4  # 1.0 turnover x 12bp, no minimum, no stamp
    assert res.costs_paid[0] == pytest.approx(expected_first, rel=1e-9)


def test_frictionless_model_still_costs_nothing():
    panel = synthetic_panel([SYM], n_days=400, seed=2)
    res = run_backtest(panel, BuyAndHold(SYM), costs=CostModel(0, 0, 0))
    assert np.all(res.costs_paid == 0.0)
    assert res.total_cost_drag == pytest.approx(0.0)
