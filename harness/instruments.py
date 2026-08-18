"""Instrument reference data — and the constraints that invalidate backtests.

Two UK facts belong in code rather than in a document, because both fail
silently and both produce a plausible, entirely fictitious equity curve:

1. UK retail investors CANNOT BUY US-domiciled ETFs. Not a broker limitation -
   regulatory, and it survived the CCI reform because the binding barrier is now
   the Overseas Funds Regime. A backtest containing SPY, VOO, QQQ or VTI is not
   optimistic. It is fiction.

2. UK stamp duty is 0.5% ON PURCHASES ONLY, and it is an attribute of the
   INSTRUMENT, never of the exchange. AIM shares, most ETFs and non-UK-
   incorporated LSE listings are exempt. Inferring it from the venue gets it
   wrong in both directions.

A documented constraint is one somebody discovers after running the backtest.
These raise at universe-construction time instead.
"""

from __future__ import annotations

from dataclasses import dataclass

SDRT_RATE = 0.005  # 0.5%, purchases only


@dataclass(frozen=True)
class InstrumentMeta:
    symbol: str
    investable_uk_retail: bool = True
    stamp_duty_rate: float = 0.0
    currency: str = "GBP"
    note: str = ""


class NotInvestable(ValueError):
    """Raised when a universe contains something UK retail cannot actually buy."""


def uk_share(symbol: str) -> InstrumentMeta:
    """UK-incorporated share. Stamp duty applies on purchase."""
    return InstrumentMeta(symbol, True, SDRT_RATE, "GBP", "UK-incorporated: SDRT on buys")


def aim_share(symbol: str) -> InstrumentMeta:
    """AIM-listed. Exempt from stamp duty."""
    return InstrumentMeta(symbol, True, 0.0, "GBP", "AIM: SDRT exempt")


def ucits_etf(symbol: str, currency: str = "GBP") -> InstrumentMeta:
    """Irish/Luxembourg-domiciled UCITS ETF. Investable, SDRT exempt.

    NOTE the sample cost: these are the *substitutes* for US-domiciled ETFs, and
    they have materially shorter histories. That silently truncates the usable
    backtest window, which is a data problem disguised as a compliance one.
    """
    return InstrumentMeta(symbol, True, 0.0, currency, "UCITS ETF: investable, SDRT exempt")


def us_domiciled_etf(symbol: str) -> InstrumentMeta:
    """US-domiciled ETF. NOT purchasable by UK retail. Present so it can be refused."""
    return InstrumentMeta(
        symbol, False, 0.0, "USD",
        "US-domiciled ETF: UK retail cannot purchase (Overseas Funds Regime)",
    )


def us_share(symbol: str) -> InstrumentMeta:
    """US-listed operating company. Investable; no UK stamp duty."""
    return InstrumentMeta(symbol, True, 0.0, "USD", "US share: no SDRT")


def synthetic(symbol: str) -> InstrumentMeta:
    """Synthetic test instrument. Investable, frictionless, not real."""
    return InstrumentMeta(symbol, True, 0.0, "GBP", "synthetic")


# Known-blocked tickers, so the common mistake fails loudly and by name.
BLOCKED_US_ETFS = ("SPY", "VOO", "QQQ", "VTI", "IVV", "VOOG", "DIA", "IWM")

DEFAULT_REGISTRY: dict[str, InstrumentMeta] = {
    s: us_domiciled_etf(s) for s in BLOCKED_US_ETFS
}


def assert_investable(
    symbols: list[str], registry: dict[str, InstrumentMeta] | None = None
) -> None:
    """Refuse a universe containing anything UK retail cannot buy.

    Call at universe construction. Unknown symbols pass - we cannot know
    everything - but the known traps fail by name with the reason attached.
    """
    reg = registry if registry is not None else DEFAULT_REGISTRY
    blocked = [
        (s, reg[s].note) for s in symbols if s in reg and not reg[s].investable_uk_retail
    ]
    if blocked:
        lines = "\n".join(f"    {s}: {note}" for s, note in blocked)
        raise NotInvestable(
            "Universe contains instruments UK retail cannot purchase.\n"
            f"{lines}\n"
            "  Any backtest on these is fiction. Use LSE-listed UCITS equivalents\n"
            "  (e.g. CSPX/VUSA for S&P 500, EQQQ for Nasdaq 100) and note that\n"
            "  their shorter histories truncate the usable sample."
        )


def meta_for(symbol: str, registry: dict[str, InstrumentMeta] | None = None) -> InstrumentMeta:
    reg = registry if registry is not None else DEFAULT_REGISTRY
    return reg.get(symbol, InstrumentMeta(symbol))
