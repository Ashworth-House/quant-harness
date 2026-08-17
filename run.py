#!/usr/bin/env python3
"""Run one strategy through the harness and print an honest report.

    python run.py --strategy ma
    python run.py --strategy ma --sizer voltarget
    python run.py --strategy buyhold --dataset noise

Every run records a trial. That is not optional, because a trial count kept by
hand is a count that flatters the person keeping it.
"""

from __future__ import annotations

import argparse

from harness import (
    BuyAndHold,
    CostModel,
    DrawdownGuard,
    FixedFraction,
    MovingAverageCrossover,
    RandomPositions,
    VolatilityTarget,
    compute_metrics,
    render_report,
    run_backtest,
    shuffled_panel,
    synthetic_panel,
)
from harness.trials import record_trial

SYM = "SYNTH"


def build_dataset(kind: str, days: int, seed: int):
    """Real data slots in here later behind the same Panel interface."""
    if kind == "trending":
        return synthetic_panel([SYM], days, seed=seed, annual_drift=0.07)
    if kind == "noise":
        # Negative control: zero drift, no structure. Nothing here is findable.
        return synthetic_panel([SYM], days, seed=seed, annual_drift=0.0)
    if kind == "shuffled":
        # Negative control: real-shaped returns, time ordering destroyed.
        return shuffled_panel(synthetic_panel([SYM], days, seed=seed, annual_drift=0.07), seed + 1)
    raise SystemExit(f"unknown dataset: {kind}")


def build_strategy(kind: str):
    if kind == "buyhold":
        return BuyAndHold(SYM)
    if kind == "ma":
        return MovingAverageCrossover(SYM, 50, 200)
    if kind == "random":
        return RandomPositions(SYM, seed=0)
    raise SystemExit(f"unknown strategy: {kind}")


def build_sizer(kind: str):
    if kind == "fixed":
        return FixedFraction(1.0)
    if kind == "voltarget":
        return VolatilityTarget(0.10, 60)
    if kind == "ddguard":
        return DrawdownGuard(VolatilityTarget(0.10, 60), 0.10, 0.25)
    raise SystemExit(f"unknown sizer: {kind}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--strategy", default="ma", choices=["buyhold", "ma", "random"])
    p.add_argument("--sizer", default="fixed", choices=["fixed", "voltarget", "ddguard"])
    p.add_argument(
        "--dataset", default="trending", choices=["trending", "noise", "shuffled"]
    )
    p.add_argument("--days", type=int, default=2500)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--no-record", action="store_true", help="do not count this trial")
    args = p.parse_args()

    panel = build_dataset(args.dataset, args.days, args.seed)
    strategy = build_strategy(args.strategy)
    sizer = build_sizer(args.sizer)
    costs = CostModel()

    result = run_backtest(panel, strategy, costs=costs, sizer=sizer)
    benchmark = run_backtest(panel, BuyAndHold(SYM), costs=costs)

    dataset_label = f"{args.dataset} (synthetic, seed={args.seed}, {args.days}d)"
    sharpe = compute_metrics(result).sharpe

    n_trials = (
        record_trial(strategy.name, sizer.name, dataset_label, sharpe)
        if not args.no_record
        else 0
    )

    print(render_report(result, benchmark, dataset_label, n_trials))

    if args.dataset in ("noise", "shuffled"):
        print()
        print("  NOTE: this dataset contains no exploitable structure by construction.")
        print("  Any apparent edge here is measuring the search, not the market.")


if __name__ == "__main__":
    main()
