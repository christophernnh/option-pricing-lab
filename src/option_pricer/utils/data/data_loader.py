# market_data_loader.py
"""
MarketDataLoader for option chains.

Features:
- Threaded expiry downloads (configurable max_workers)
- Retry + exponential backoff for flaky network calls
- Optional non-destructive liquidity tagging (is_liquid flag)
- Configurable liquidity filters (min OI, min volume, max spread pct, stale-last policy)
- Attaches spot to OptionChain and computes derived fields
- Clear extension point for computing implied vol / vega (placeholder)
- Streamlit cache decorator (commented) ready
"""

from __future__ import annotations

import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime
from typing import List, Optional, Dict, Any, Tuple, Callable

import yfinance as yf

from src.option_pricer.models.option import OptionContract, OptionChain

"""Set up logging."""
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


@dataclass
class LoaderConfig:
    """Configuration for MarketDataLoader behavior."""
    min_open_interest: int = 10
    min_volume: int = 1
    max_spread_pct: float = 0.25       # e.g., (ask - bid) / bid <= 0.25
    ignore_stale_last: bool = False     # if True, require bid & ask to compute mid; otherwise allow last
    
    max_workers: int = 8
    retries: int = 3
    backoff_factor: float = 0.8        # exponential backoff multiplier (seconds)

class MarketDataLoader:
    """
    Robust market data loader for option chains using yfinance.
    Usage:
        cfg = LoaderConfig(...)
        loader = MarketDataLoader(cfg)
        chain = loader.get_option_chain("AAPL", filter=True, tag_liquidity=False)
    """

    def __init__(self, config: Optional[LoaderConfig] = None):
        self.config = config or LoaderConfig()

    # To cache results if function is ran using the same input:
    # @st.cache_data(ttl=60*10)
    def get_option_chain(self, ticker: str, filter: bool = True, tag_liquidity: bool = False) -> OptionChain:
        """
        Download the option chain for `ticker` and return an OptionChain.

        :param ticker: ticker symbol, e.g. "AAPL"
        :param filter: if True, apply liquidity filters (destructive: removes illiquid contracts)
        :param tag_liquidity: if True, keep all contracts but set attribute `is_liquid` on contracts
                               NOTE: if both filter=True and tag_liquidity=True, filtering is applied
                               and tagging is performed on the filtered results.
        """
        tk = yf.Ticker(ticker)                                 # yfinance Ticker object
        as_of_date = datetime.utcnow().date().isoformat()      # record current snapshot date

        expiries: List[str] = list(getattr(tk, "options", []) or [])        # list of expiry dates
        logger.info("Ticker %s: found %d expiries", ticker, len(expiries))  # log expiry count

        spot = self._safe_get_spot_with_retry(tk)                     # attempt to get spot price with retries
        if spot is None:
            logger.warning("Ticker %s: could not retrieve spot price; moneyness will be None", ticker)

        contracts: List[OptionContract] = []
        if expiries:
            # Use multithreading to load each expiry in parallel
            with ThreadPoolExecutor(max_workers=self.config.max_workers) as ex:
                future_to_expiry = {}   # Creates a dictionary with type: Dict[Any, str]  
                # Loop through all expiry load tasks
                for expiry in expiries:
                    future = ex.submit(self._load_single_expiry_with_retry, tk, expiry, spot) # Fetch call and put prices concurrently
                    future_to_expiry[future] = expiry                                         # Map future to expiry date

                # Wait for all futures to complete using as_completed
                for fut in as_completed(future_to_expiry):
                    expiry = future_to_expiry[fut]          # Get the expiry date for this future
                    try:
                        loaded = fut.result()               # Get the call and put result from yfinance
                        contracts.extend(loaded)            # Insert loaded contract list into a bigger list (2D list)
                        logger.info("Ticker %s: loaded %d contracts for expiry %s", ticker, len(loaded), expiry)
                    except Exception as e:
                        logger.exception("Ticker %s: failed loading expiry %s: %s", ticker, expiry, e)

        # Sort contracts for deterministic ordering
        contracts = sorted(contracts, key=lambda c: (c.expiry, c.strike, c.option_type))

        chain = OptionChain(underlying=ticker, as_of=as_of_date, spot=spot, contracts=contracts)

        # Calculate derived fields (mid, moneyness, maturity)
        chain.enrich()

        # Tag liquidity (non-destructive) if requested
        if tag_liquidity:
            self._tag_liquidity(chain)

        # Filter (destructive) if requested
        if filter:
            chain = self._filter_chain(chain)

        logger.info("Ticker %s: returning chain with %d contracts", ticker, len(chain.contracts))
        return chain

    # ---------------------------
    # Retry & network helpers
    # ---------------------------
    def _retry_loop(self, func: Callable[..., Any], *args, **kwargs) -> Any:
        """Simple retry loop with exponential backoff."""
        retries = max(0, int(self.config.retries))
        backoff = float(self.config.backoff_factor) or 0.5
        last_exc = None
        for attempt in range(retries + 1):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                last_exc = e
                wait = backoff * (2 ** attempt)
                logger.debug("Attempt %d failed with %s — retrying in %.2fs", attempt + 1, e, wait)
                time.sleep(wait)
        # All retries failed
        logger.exception("All %d retries failed for function %s", retries, getattr(func, "__name__", str(func)))
        raise last_exc

    def _safe_get_spot_with_retry(self, tk: yf.Ticker) -> Optional[float]:
        """Wrapper around _safe_get_spot using retry logic."""
        try:
            return self._retry_loop(self._safe_get_spot, tk)
        except Exception:
            return None

    def _load_single_expiry_with_retry(self, tk: yf.Ticker, expiry: str, spot: Optional[float]) -> List[OptionContract]:
        """Load one expiry with retry logic around yf.option_chain."""
        return self._retry_loop(self._load_single_expiry, tk, expiry, spot)

    # ---------------------------
    # Core loading
    # ---------------------------
    def _load_single_expiry(self, tk: yf.Ticker, expiry: str, spot: Optional[float]) -> List[OptionContract]:
        """Load one expiry's option chain and convert to OptionContract list."""
        try:
            df_chain = tk.option_chain(expiry)
        except Exception as e:
            logger.exception("yf.option_chain failed for expiry %s: %s", expiry, e)
            return []

        result: List[OptionContract] = []
        # Calls and puts - keep code DRY by delegating to _build_contracts
        result.extend(self._build_contracts(df_chain.calls, "C", tk.ticker, expiry, spot))
        result.extend(self._build_contracts(df_chain.puts, "P", tk.ticker, expiry, spot))
        return result

    def _build_contracts(self, df, opt_type: str, underlying: str, expiry: str, spot: Optional[float]) -> List[OptionContract]:
        """
        Convert yfinance DataFrame (calls or puts) to OptionContract objects.
        This method is resilient to schema differences across yfinance versions.
        """
        out: List[OptionContract] = []
        for row in df.itertuples(index=False):
            # Extract symbol and strike robustly
            try:
                symbol = getattr(row, "contractSymbol", None) or getattr(row, "symbol", None)
                strike_raw = getattr(row, "strike", None)
                if strike_raw is None:
                    logger.debug("Skipping row with missing strike: %r", row)
                    continue
                strike = float(strike_raw)
            except Exception:
                logger.debug("Skipping invalid row: %r", row)
                continue

            bid = self._to_float(getattr(row, "bid", None))
            ask = self._to_float(getattr(row, "ask", None))
            last = self._to_float(getattr(row, "lastPrice", None)) or self._to_float(getattr(row, "last", None))
            volume = self._to_int(getattr(row, "volume", None))
            open_interest = self._to_int(getattr(row, "openInterest", None))

            c = OptionContract(
                symbol=symbol,
                underlying=underlying,
                expiry=expiry,
                strike=strike,
                option_type=opt_type,
                bid=bid,
                ask=ask,
                last=last,
                volume=volume,
                open_interest=open_interest,
            )

            # Derived fields that don't require heavy computation
            c.compute_mid()
            self._annotate_maturity(c, spot)

            # Placeholder: compute implied vol & vega if you have a BSM pricer available
            # try:
            #     if c.market_price is not None and c.maturity_years and c.maturity_years > 0 and spot:
            #         c.implied_vol = compute_implied_vol(c.market_price, spot, c.strike, c.maturity_years, ...)
            #         c.vega = compute_vega(spot, c.strike, c.maturity_years, c.implied_vol, ...)
            # except Exception:
            #     logger.debug("IV compute failed for %s", c.symbol)

            out.append(c)
        return out

    # ---------------------------
    # Conversion helpers
    # ---------------------------
    @staticmethod
    def _to_float(v: Any) -> Optional[float]:
        try:
            if v is None:
                return None
            val = float(v)
            if val < 0:
                return None
            return val
        except Exception:
            return None

    @staticmethod
    def _to_int(v: Any) -> Optional[int]:
        try:
            if v is None:
                return None
            val = int(v)
            if val < 0:
                return None
            return val
        except Exception:
            return None

    # ---------------------------
    # Spot / maturity helpers
    # ---------------------------
    @staticmethod
    def _safe_get_spot(tk: yf.Ticker) -> Optional[float]:
        """Get latest close price from tk.history, return None on failure (no retries)."""
        hist = tk.history(period="1d")
        if hist is None or hist.empty:
            raise ValueError("Empty history")
        return float(hist["Close"].iloc[-1])

    @staticmethod
    def _annotate_maturity(contract: OptionContract, spot: Optional[float]) -> None:
        """Compute moneyness and maturity_years in-place (if possible)."""
        try:
            if spot is not None and spot > 0:
                contract.moneyness = contract.strike / spot
        except Exception:
            contract.moneyness = None

        try:
            expiry_dt = datetime.strptime(contract.expiry, "%Y-%m-%d").date()
            today = datetime.utcnow().date()
            days = max((expiry_dt - today).days, 0)
            contract.maturity_years = days / 365.0 if days > 0 else 0.0
        except Exception:
            contract.maturity_years = None

    # ---------------------------
    # Liquidity tagging & filtering
    # ---------------------------
    def _is_liquid(self, c: OptionContract) -> bool:
        """Return True if contract meets liquidity criteria defined in config."""
        oi = c.open_interest or 0
        vol = c.volume or 0
        if oi < self.config.min_open_interest or vol < self.config.min_volume:
            return False

        # spread percentage check
        if c.bid is not None and c.ask is not None and c.bid > 0:
            spread_pct = (c.ask - c.bid) / c.bid
            if spread_pct > self.config.max_spread_pct:
                return False

        # If mid is none, enforce it
        if c.mid is None:
            return False

        return True

    def _tag_liquidity(self, chain: OptionChain) -> None:
        """
        Non-destructively tag contracts with attribute `is_liquid` (True/False).
        This leaves the original chain intact but adds metadata useful for UI.
        """
        for c in chain.contracts:
            setattr(c, "is_liquid", self._is_liquid(c))

    def _filter_chain(self, chain: OptionChain) -> OptionChain:
        """Return a new OptionChain with only liquid contracts (destructive)."""
        filtered = [c for c in chain.contracts if self._is_liquid(c)]
        return OptionChain(underlying=chain.underlying, as_of=chain.as_of, spot=chain.spot, contracts=filtered)
