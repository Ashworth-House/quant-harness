# Pre-registration log

Borrowed from clinical trials. **The hypothesis, the reasoning and the success
criterion are written and committed here BEFORE the backtest is run.** Git
supplies the timestamp, which is what makes the claim credible rather than
merely asserted.

Rules, and they only work if they are not bent:

1. Nothing is added retrospectively. If it was not committed before the run, it
   was not pre-registered.
2. The success criterion is stated numerically, in advance.
3. **Rejected hypotheses stay in this file.** They are the deliverable. A log
   containing only successes is a log that has been edited.
4. If a hypothesis is amended after seeing results, that is recorded as an
   amendment with a date and a reason — not a rewrite.

---

## H1 — The moving-average crossover underperforms buy-and-hold after costs

**Registered:** 17 August 2026, before first run
**Status:** RESOLVED — hypothesis supported

**Hypothesis.** A 50/200-day moving-average crossover on a positively-drifting
series will produce a lower total return than buy-and-hold once realistic
retail costs (5bp commission, 5bp half-spread, 2bp slippage) are applied.

**Reasoning.** The strategy is out of the market for part of a rising series and
pays costs on every crossing. Both effects are negative in expectation against a
drifting benchmark.

**Success criterion.** Mean total return across 12 independent synthetic seeds
is lower for the crossover than for buy-and-hold, and it underperforms on at
least 7 of the 12.

**Result.** Supported. Both criteria met — see
`tests/test_harness.py::test_moving_average_crossover_loses_to_buy_and_hold`,
which encodes the criterion and passes.

**Note.** On the single seed used by `run.py --dataset trending --seed 42` the
crossover has a *higher* CAGR than buy-and-hold and a *worse* Sharpe. That is
the reason the criterion was specified across twelve seeds rather than one, and
it is a small live demonstration of why single-backtest conclusions are unsafe.

---

## H2 — [next hypothesis goes here, before it is tested]

**Registered:**
**Status:** NOT YET RUN

**Hypothesis.**

**Reasoning.**

**Success criterion.**

**Result.**
