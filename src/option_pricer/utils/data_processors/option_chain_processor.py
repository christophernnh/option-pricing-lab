"""
Option Chain Processor
-----------------------------

Takes raw option chain data from loader
and computes:
- Mid price
- Implied volatility
- Greeks (optional)

Returns a list of OptionPoint dataclasses (to surface builder for heatmapping).

"""

from dataclasses import dataclass
from typing import Literal, List, Any, Optional

from src.option_pricer.utils.pricers.black_scholes import (
    bsm_delta,
    bsm_gamma,
    bsm_theta,
    bsm_rho,
)
from src.option_pricer.utils.pricers.implied_volatility import implied_volatility


OptionType = Literal["C", "P"]


@dataclass
class OptionPoint:
    symbol: str
    expiry: str
    strike: float
    type: OptionType
    bid: float
    ask: float
    mid: float
    implied_vol: Optional[float]
    delta: Optional[float]
    gamma: Optional[float]
    theta: Optional[float]
    rho: Optional[float]


class OptionChainProcessor:
    def __init__(self, risk_free_rate: float = 0.05, dividend_yield: float = 0.00):
        self.r = risk_free_rate
        self.q = dividend_yield

    def process_chain(self, raw_chain: List[dict], spot: float, tau: float) -> List[OptionPoint]:
        """
        Main entrypoint.

        raw_chain format expected (example):
        [
            {
                "symbol": "AAPL",
                "expiry": "2025-01-17",
                "strike": 150,
                "type": "C",
                "bid": 3.25,
                "ask": 3.55
            },
            ...
        ]
        """

        processed: List[OptionPoint] = []

        for row in raw_chain:
            bid = row["bid"]
            ask = row["ask"]
            mid = (bid + ask) / 2 if (bid and ask) else None

            # 1. Compute Implied Vol
            iv = None
            if mid and mid > 0:
                try:
                    iv = implied_volatility(
                        price=mid,
                        S=spot,
                        K=row["strike"],
                        r=self.r,
                        q=self.q,
                        tau=tau,
                        option_type=row["type"],
                    )
                except Exception:
                    iv = None

            # 2. Greeks (only compute if IV succeeded)
            if iv:
                delta = bsm_delta(spot, row["strike"], self.r, self.q, iv, tau, row["type"])
                gamma = bsm_gamma(spot, row["strike"], self.r, self.q, iv, tau)
                theta = bsm_theta(spot, row["strike"], self.r, self.q, iv, tau, row["type"])
                rho = bsm_rho(spot, row["strike"], self.r, self.q, iv, tau, row["type"])
            else:
                delta = gamma = theta = rho = None

            processed.append(
                OptionPoint(
                    symbol=row["symbol"],
                    expiry=row["expiry"],
                    strike=row["strike"],
                    type=row["type"],
                    bid=bid,
                    ask=ask,
                    mid=mid,
                    implied_vol=iv,
                    delta=delta,
                    gamma=gamma,
                    theta=theta,
                    rho=rho,
                )
            )

        return processed
