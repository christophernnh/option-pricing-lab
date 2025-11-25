# data_processors/iv_surface_builder.py

import pandas as pd
from typing import List
from src.option_pricer.models.option import OptionContract

class IVSurfaceBuilder:
    """
    Builds implied volatility surfaces (call and put)
    as strike x expiry matrices.
    """

    @staticmethod
    def to_dataframe(contracts: List[OptionContract]) -> pd.DataFrame:
        """Convert list of OptionContract objects into a flat DataFrame.

        Uses `maturity_years` from each contract to ensure time is in years.
        """
        rows = []
        for c in contracts:
            rows.append(
                {
                    "expiry": c.expiry,
                    "maturity_years": c.maturity_years,
                    "strike": c.strike,
                    "iv": c.implied_vol,
                    "type": c.option_type,
                }
            )
        return pd.DataFrame(rows)

    @staticmethod
    def build_iv_surfaces(df: pd.DataFrame):
        """Build two IV surfaces: one for CALLS and one for PUTS.

        Expects columns: ['strike', 'maturity_years', 'type', 'iv'].
        """

        # type cleaning
        df["strike"] = pd.to_numeric(df["strike"], errors="coerce")
        df["maturity_years"] = pd.to_numeric(df["maturity_years"], errors="coerce")
        df["iv"] = pd.to_numeric(df["iv"], errors="coerce")

        df = df.dropna(subset=["strike", "maturity_years", "iv", "type"])

        # split calls / puts (OptionContract uses "C" / "P")
        calls = df[df["type"].str.upper() == "C"]
        puts = df[df["type"].str.upper() == "P"]

        # pivot matrices: strike x maturity (years)
        call_surface = (
            calls.pivot_table(
                index="strike",
                columns="maturity_years",
                values="iv",
                aggfunc="mean",
            )
            .sort_index(axis=0)
            .sort_index(axis=1)
            if not calls.empty
            else pd.DataFrame()
        )

        put_surface = (
            puts.pivot_table(
                index="strike",
                columns="maturity_years",
                values="iv",
                aggfunc="mean",
            )
            .sort_index(axis=0)
            .sort_index(axis=1)
            if not puts.empty
            else pd.DataFrame()
        )

        return call_surface, put_surface
