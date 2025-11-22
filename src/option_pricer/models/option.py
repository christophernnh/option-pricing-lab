from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, List, Dict
import pandas as pd


@dataclass
class OptionContract:
    """
    Represents one option contract (one strike, one expiry, one type).
    """
    symbol: str                  # exchange ticker used by exchanges/brokers (e.g. "AAPL230616C00150000")
    underlying: str              # underlying asset ticker (e.g. "AAPL")
    expiry: str                  # "YYYY-MM-DD"
    strike: float                # strike price of the option
    option_type: str             # "C"(call) or "P"(put)

    # Market data
    bid: Optional[float]         # Highest price buyer is willing to pay
    ask: Optional[float]         # Lowest price seller is willing to accept
    last: Optional[float]        # Last transaction price the option traded at
    volume: Optional[int]        # Number of contracts traded today
    open_interest: Optional[int] # Total number of outstanding contracts that have not been settled

    # Derived fields
    mid: Optional[float] = None             # Price between bid/ask
    implied_vol: Optional[float] = None     # Implied volatility by reversing BSM from market price
    vega: Optional[float] = None            # Sensitivity of option price to changes in implied volatility
    moneyness: Optional[float] = None       # Strike / Spot (use ln(K/S) for some models)
    maturity_years: Optional[float] = None  # Time to expiry in years

    # -------------------------------------------------
    def compute_mid(self) -> float:
        """
        Computes mid price between bid/ask, falling back to last trade if necessary.
        A neutral estimate of fair market value
        """
        if self.bid is not None and self.ask is not None:
            self.mid = (self.bid + self.ask) / 2.0
        elif self.last is not None:
            self.mid = self.last
        else:
            self.mid = None
        return self.mid

    # -------------------------------------------------
    def compute_moneyness(self, spot: float) -> float:
        """
        Computes strike(K) / spot(S) to determine ATM or OTM status.
        If K / S < 1, then Strike < Spot → Call is in the money, Put is out of the money
        If K / S = 1, then Strike equals spot → ATM
        If K / S > 1, then Strike > Spot → Call is out of the money, Put is in the money
        """
        if spot is not None and spot > 0:
            self.moneyness = self.strike / spot
        return self.moneyness

    # -------------------------------------------------
    def compute_maturity(self, as_of_date: str) -> float:
        """
        Computes time to maturity in years: (expiry - today) / 365.0
        Time to maturity in years is used in Black-Scholes and other models.
        """
        try:
            expiry_dt = datetime.strptime(self.expiry, "%Y-%m-%d").date()
            today = datetime.strptime(as_of_date, "%Y-%m-%d").date()
            days = max((expiry_dt - today).days, 0)
            self.maturity_years = days / 365.0
        except Exception:
            self.maturity_years = None
        return self.maturity_years

    # -------------------------------------------------
    @property
    def market_price(self) -> Optional[float]:
        """
        Return mid if available, else last price.
        """
        return self.mid if self.mid is not None else self.last

@dataclass
class OptionChain:
    """
    Represents a full option chain (all expiries, all strikes).
    """
    underlying: str
    as_of: str
    spot: Optional[float] = None
    contracts: List[OptionContract] = field(default_factory=list)

    # -------------------------------------------------
    def enrich(self):
        """
        Computes all derived fields for all contracts.
        """
        for c in self.contracts:
            c.compute_mid()
            c.compute_moneyness(self.spot)
            c.compute_maturity(self.as_of)

    # -------------------------------------------------
    def by_expiry(self) -> Dict[str, List[OptionContract]]:
        """
        Groups contracts by expiry date.
        """
        out: Dict[str, List[OptionContract]] = {}
        for c in self.contracts:
            out.setdefault(c.expiry, []).append(c)
        return out

    # -------------------------------------------------
    def filter_liquid(
        self,
        min_oi: int = 10,
        min_volume: int = 1
    ) -> "OptionChain":
        """
        Returns a new OptionChain containing only contracts
        with reasonable liquidity.
        """
        filtered = []
        for c in self.contracts:
            oi = c.open_interest if c.open_interest is not None else 0
            vol = c.volume if c.volume is not None else 0
            if oi >= min_oi and vol >= min_volume:
                filtered.append(c)

        return OptionChain(
            underlying=self.underlying,
            as_of=self.as_of,
            spot=self.spot,
            contracts=filtered,
        )

    # -------------------------------------------------
    def to_dataframe(self) -> pd.DataFrame:
        """
        Converts all contracts into a Pandas DataFrame.
        """
        rows = []
        for c in self.contracts:
            rows.append({
                "symbol": c.symbol,
                "underlying": c.underlying,
                "expiry": c.expiry,
                "strike": c.strike,
                "type": c.option_type,
                "bid": c.bid,
                "ask": c.ask,
                "mid": c.mid,
                "last": c.last,
                "volume": c.volume,
                "open_interest": c.open_interest,
                "iv": c.implied_vol,
                "vega": c.vega,
                "moneyness": c.moneyness,
                "maturity": c.maturity_years,
            })
        return pd.DataFrame(rows)

    # -------------------------------------------------
    def expiries(self) -> List[str]:
        """
        Returns all unique expiry dates.
        """
        return sorted({c.expiry for c in self.contracts})
