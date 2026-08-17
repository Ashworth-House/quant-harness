# Quant Harness

**An evaluation harness for trading strategies. The goal of v1 is not a profitable
strategy — it is a scoreboard trustworthy enough to prove a bad one is bad.**

The reason is a number. Wiecki, Campbell, Lent & Stauth studied 888 algorithmic
strategies built on Quantopian, each with at least six months of live
out-of-sample results, and found that **backtest Sharpe ratio predicted
out-of-sample Sharpe with R² < 0.025** — essentially nothing. They also found
that the more backtesting a quant had done, the *larger* the gap between
backtest and reality.

So the strategy is not the hard part. Knowing whether it works is.

---

## Quick start

```bash
python3 -m venv .venv
./.venv/bin/pip install -r requirements.txt
./.venv/bin/python -m pytest          # 25 passed, 1 skipped
./.venv/bin/python run.py --strategy ma --dataset trending
```

Try the negative control, where nothing is findable by construction:

```bash
./.venv/bin/python run.py --strategy ma --dataset noise
```

---

## Why synthetic data

The harness is developed and tested against synthetic series on purpose:

- **No licence.** Nothing here can breach a vendor's terms, so the repository
  can be public from day one.
- **No survivorship bias.** There is no universe to have deleted the failures from.
- **Known ground truth.** A zero-drift random walk contains no exploitable
  structure, so anything a strategy "finds" there is measuring the search rather
  than the market. That is the negative control, free.

Real data slots in behind `Panel` — write an adapter, change nothing else.
**No licensed market data is ever committed to this repository.**

---

## What is enforced, and where

| Guarantee | Enforced by |
|---|---|
| A strategy cannot see the future | `MarketWindow` slices to `t`; `assert_no_lookahead` corrupts everything after `t` and requires identical output |
| Fills happen at the next bar's open, never the signal's close | `engine.run_backtest`, pinned by `test_fills_use_next_bar_open_not_signal_close` |
| Costs are always paid and never netted silently | `CostModel`, reported as explicit drag |
| No metric appears without its benchmark | `report.render_report` |
| Every run is counted | `trials.record_trial` — automatic, because a hand-kept count flatters its keeper |
| No leverage in v1 | Engine raises above 1.0 gross exposure |

The lookahead detector is tested against a deliberate cheat
(`test_the_lookahead_detector_actually_catches_a_cheat`). **A detector that never
fires proves nothing.**

---

## Layout

```
harness/
  data.py       Panel, synthetic generator, shuffled negative control
  window.py     MarketWindow — the no-lookahead boundary
  strategy.py   Strategy protocol + buy-and-hold, MA crossover, random
  sizing.py     Fixed, volatility-targeted, drawdown-guarded  [M3b]
  engine.py     Costs, next-bar fills, the loop
  metrics.py    Sharpe, Sortino, drawdown, Calmar, cost drag
  trials.py     Trial counting and the deflation warning
  report.py     Rendering
  nulldist.py   [M6] NOT IMPLEMENTED — see below
tests/
  test_harness.py               25 tests, all passing
  test_m6_null_distribution.py  the specification for M6, skipped
```

---

## The next milestone is deliberately unbuilt

`harness/nulldist.py` is a stub. **M6 — the random-strategy null distribution —
belongs to the co-founder**, because it is the piece that will be defended in an
interview and it should be built by the person defending it.

Run a coin-flip strategy a thousand times. Plot the distribution of Sharpe
ratios. That is what luck looks like on this data, with these costs, over this
period. Then put a real strategy on the same chart: if it sits inside the random
cloud, it has no demonstrated edge, however good its equity curve looks alone.

`tests/test_m6_null_distribution.py` is the brief. Delete its `pytest.skip` and
make the tests pass.

---

## Roadmap

| | | Status |
|---|---|---|
| M0–M1 | Repo, tests, data layer | ✅ |
| M2 | Strategy interface, buy-and-hold reproduces benchmark | ✅ |
| M3 | Execution simulator: costs, next-bar fills | ✅ |
| M3b | Position sizing as a measurable variable | ✅ |
| M4 | Metrics and one-command report | ✅ |
| M5 | MA crossover visibly loses after costs | ✅ |
| **M6** | **Random null distribution** | **Stub — co-founder** |
| **K1–K3** | **Known-answer tests — validate the instrument** | **Next, and before any real hypothesis** |
| M7 | Walk-forward with purging and embargo | Not started |
| M8 | Deflated Sharpe using the trial count | Not started |
| M9 | Pre-registration workflow | Log started |
| M10 | Cross-validate against an established backtester | Not started |

Until M6, M7 and M8 exist, **every report says so on its face**. A Sharpe ratio
with no null distribution behind it is not evidence, and the report refuses to
pretend otherwise.

---

## Known-answer tests (K1–K3)

Before any real hypothesis, run three strategies whose answer the literature
already knows. **A harness that reports positive net alpha for weekly short-term
reversal is a harness that will approve anything you feed it.**

| | Test | Expected | Diagnoses |
|---|---|---|---|
| K1 | Volatility-managed exposure beats unmanaged out-of-sample | **REJECT** | Walk-forward leaking in-sample information |
| K2 | Weekly short-term reversal earns net alpha in large caps | **REJECT** | Cost model not wired into the P&L — the costs alone are ~9.6%/yr |
| K3 | Long-only 12-1 momentum earns >1%/yr alpha in large caps | REJECT / trivial | Point-in-time constituent handling; survivorship bias inflates this one dramatically |

Validate the instrument before trusting the measurement.

---

## What the harness cannot do

At a true Sharpe of 0.5 — a *good* outcome — reaching t = 2 takes ~18 years of
live data, and ~46 years once corrected for 50 trials tested (Lo, 2002).

**So this harness can only ever reject a strategy, never confirm one.** That is
still worth building: false positives are the documented failure mode, and
rejection is what protects against them. But absence of rejection is not
confirmation, and the decision to risk money has to rest on economic mechanism —
*why does this money exist, who is on the other side, and why can't they stop?* —
which no amount of backtesting can answer.
