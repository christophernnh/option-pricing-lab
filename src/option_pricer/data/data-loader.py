import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from typing import List, Optional

import yfinance as yf


# adjust the imports to match your package structure
from option_pricer.models.option import OptionContract
from option_pricer.models.option import OptionChain


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class MarketDataLoader:
    def __init__(self, min_open_interest: int = 0, min_volume: int = 0, max_workers: int = 8):
        """
        :param min_open_interest: minimal open interest for a contract to be considered 'liquid'
        :param min_volume: minimal volume for a contract to be considered 'liquid'
        :param max_workers: number of threads to use when loading expiries in parallel
        """
        self.min_open_interest = int(min_open_interest or 0)
        self.min_volume = int(min_volume or 0)
        self.max_workers = max_workers

    # If using Streamlit, you can enable caching here:
    # @st.cache_data(ttl=60*10)  # cache for 10 minutes (example)
    def get_option_chain(self, ticker: str) -> OptionChain:
        """Public entry: download option chain for `ticker` and return an enriched OptionChain."""
        tk = yf.Ticker(ticker)
        as_of_date = datetime.utcnow().date().isoformat()

        # fetch expiries; fallback to empty list
        expiries: List[str] = list(getattr(tk, "options", []) or [])
        logger.info("Ticker %s has %d expiries", ticker, len(expiries))

        # fetch spot price safely
        spot = self._safe_get_spot(tk)
        if spot is None:
            logger.warning("Failed to fetch spot price for %s; some fields (moneyness) may be None", ticker)

        # Concurrently load expiry chains
        contracts: List[OptionContract] = []
        if expiries:
            with ThreadPoolExecutor(max_workers=self.max_workers) as ex:
                futures = {ex.submit(self._load_single_expiry, tk, expiry, spot): expiry for expiry in expiries}
                for fut in as_completed(futures):
                    expiry = futures[fut]
                    try:
                        loaded = fut.result()
                        contracts.extend(loaded)
                        logger.info("Loaded %d contracts for expiry %s", len(loaded), expiry)
                    except Exception as e:
                        logger.exception("Failed to load expiry %s: %s", expiry, e)

        # build OptionChain and apply filters
        chain_obj = OptionChain(underlying=ticker, as_of=as_of_date, contracts=sorted(contracts, key=lambda c: (c.expiry, c.strike, c.option_type)))
        chain_obj.spot = spot  # attach spot for downstream use
        if self.min_open_interest or self.min_volume:
            chain_obj = chain_obj.filter_liquid(self.min_open_interest, self.min_volume)
            logger.info("After filtering, %d contracts remain", len(chain_obj.contracts))

        return chain_obj

    def _load_single_expiry(self, tk: yf.Ticker, expiry: str, spot: Optional[float]) -> List[OptionContract]:
        """
        Load one expiry's calls and puts, convert rows to OptionContract objects,
        and annotate derived fields.
        """
        try:
            chain = tk.option_chain(expiry)
        except Exception as e:
            logger.exception("yf.option_chain failed for expiry %s: %s", expiry, e)
            return []

        contracts: List[OptionContract] = []
        # reuse helper for calls and puts
        contracts.extend(self._build_contracts(chain.calls, 'C', tk.ticker, expiry, spot))
        contracts.extend(self._build_contracts(chain.puts, 'P', tk.ticker, expiry, spot))
        return contracts

    def _build_contracts(self, df, opt_type: str, underlying: str, expiry: str, spot: Optional[float]) -> List[OptionContract]:
        """Convert a DataFrame of options (calls or puts) into OptionContract objects."""
        result: List[OptionContract] = []
        for row in df.itertuples(index=False):
            # safe extraction with getattr fallback to keep compatibility across yfinance versions
            try:
                symbol = getattr(row, "contractSymbol", None) or getattr(row, "symbol", None)
                strike = float(getattr(row, "strike"))
            except Exception:
                logger.debug("Skipping row due to missing symbol/strike: %r", row)
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

            # derived fields
            c.compute_mid()
            self._annotate_maturity(c, spot)
            # placeholder: you can compute implied_vol and vega here if you want
            # e.g. c.implied_vol = compute_implied_vol(c.market_price, spot, ...)
            result.append(c)
        return result

    @staticmethod
    def _to_float(v) -> Optional[float]:
        """Safely convert value to float; treat negative or missing as None."""
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
    def _to_int(v) -> Optional[int]:
        """Safely convert value to int; treat negative or missing as None."""
        try:
            if v is None:
                return None
            val = int(v)
            if val < 0:
                return None
            return val
        except Exception:
            return None

    @staticmethod
    def _safe_get_spot(tk: yf.Ticker) -> Optional[float]:
        """Get latest close price from tk.history, return None on failure."""
        try:
            hist = tk.history(period="1d")
            if hist is None or hist.empty:
                return None
            return float(hist["Close"].iloc[-1])
        except Exception:
            logger.exception("Failed to fetch history/spot for ticker %s", getattr(tk, "ticker", "unknown"))
            return None

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
