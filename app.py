import streamlit as st
import pandas as pd

from src.option_pricer.utils.data.data_loader import LoaderConfig, MarketDataLoader
from src.option_pricer.utils.data_processors.option_chain_processor import OptionChainProcessor
from src.option_pricer.utils.data_processors.iv_surface_builder import IVSurfaceBuilder
from src.option_pricer.utils.data_processors.iv_surface_plot import plot_iv_surface


def main() -> None:
	"""Single-page Streamlit app for IV surface exploration."""

	st.set_page_config(page_title="Option IV Surface Explorer", layout="wide")
	st.title("Option Implied Volatility Surface Explorer")

	# Sidebar configuration
	st.sidebar.header("Configuration")

	ticker = st.sidebar.text_input(
		"Ticker", value="AAPL", help="Underlying symbol to fetch from Yahoo Finance."
	)

	risk_free_rate = st.sidebar.number_input(
		"Risk-free rate (annual)", value=0.05, step=0.01, format="%.4f"
	)
	dividend_yield = st.sidebar.number_input(
		"Dividend yield (annual)", value=0.00, step=0.01, format="%.4f"
	)

	min_open_interest = st.sidebar.number_input(
		"Min open interest", min_value=0, value=10, step=10
	)
	min_volume = st.sidebar.number_input(
		"Min volume", min_value=0, value=1, step=1
	)
	max_spread_pct = st.sidebar.slider(
		"Max bid-ask spread %", min_value=0.0, max_value=1.0, value=0.25, step=0.05
	)

	option_type_filter = st.sidebar.selectbox(
		"Option type", options=["All", "Calls", "Puts"], index=0
	)

	build_button = st.sidebar.button("Build IV Surface")

	if not build_button:
		st.info("Configure parameters in the sidebar, then click 'Build IV Surface'.")
		return

	if not ticker:
		st.error("Please enter a valid ticker symbol.")
		return

	with st.spinner("Loading option chain and building IV surface..."):
		# 1) Load option chain
		loader_cfg = LoaderConfig(
			min_open_interest=int(min_open_interest),
			min_volume=int(min_volume),
			max_spread_pct=float(max_spread_pct),
		)
		loader = MarketDataLoader(loader_cfg)
		chain = loader.get_option_chain(ticker=ticker, filter=True, tag_liquidity=False)

		if not chain.contracts:
			st.warning("No liquid option contracts returned for this configuration.")
			return

		# 2) Process contracts into OptionPoint objects with IVs
		processor = OptionChainProcessor(risk_free_rate=risk_free_rate, dividend_yield=dividend_yield)

		raw_chain: list[dict] = []
		spot = chain.spot or 0.0
		for c in chain.contracts:
			# Use per-contract maturity (in years) as tau
			tau = c.maturity_years or 0.0
			if tau <= 0:
				continue

			# Respect option type filter
			if option_type_filter == "Calls" and c.option_type != "C":
				continue
			if option_type_filter == "Puts" and c.option_type != "P":
				continue

			raw_chain.append(
				{
					"symbol": c.symbol,
					"expiry": c.expiry,
					"strike": c.strike,
					"type": c.option_type,
					"bid": c.bid or 0.0,
					"ask": c.ask or 0.0,
				}
			)

		if not raw_chain:
			st.warning("No contracts passed the filters/maturity check to compute IV.")
			return

		# OptionChainProcessor expects a single tau, but our options have varying maturities.
		# We call it per-expiry to keep tau consistent within each batch.
		points = []
		df_raw = pd.DataFrame(raw_chain)
		for expiry, df_group in df_raw.groupby("expiry"):
			sample_contracts = [c for c in chain.contracts if c.expiry == expiry]
			if not sample_contracts:
				continue
			tau = sample_contracts[0].maturity_years or 0.0
			if tau <= 0:
				continue

			group_list = df_group.to_dict(orient="records")
			points.extend(processor.process_chain(group_list, spot=spot, tau=tau))

		if not points:
			st.warning("Implied volatilities could not be computed for any contracts.")
			return

		# 3) Display full OptionChainProcessor results
		st.subheader("Processed Option Chain (OptionChainProcessor output)")
		points_df = pd.DataFrame(
			[
				{
					"symbol": p.symbol,
					"expiry": p.expiry,
					"strike": p.strike,
					"type": p.type,
					"bid": p.bid,
					"ask": p.ask,
					"mid": p.mid,
					"iv": p.implied_vol,
					"delta": p.delta,
					"gamma": p.gamma,
					"theta": p.theta,
					"rho": p.rho,
				}
				for p in points
			]
		)
		st.dataframe(points_df)

		# 4) Build IV surfaces for calls and puts from OptionPoint list
		# Map OptionPoint -> minimal OptionContract-like objects, including maturity_years in years
		contracts_like = []
		for p in points:
			# find matching original contract to grab maturity_years in years
			orig = next((c for c in chain.contracts if c.expiry == p.expiry and c.strike == p.strike and c.option_type == p.type), None)
			maturity_years = orig.maturity_years if orig is not None else None

			contracts_like.append(
				type("_TmpContract", (), {
					"expiry": p.expiry,
					"maturity_years": maturity_years,
					"strike": p.strike,
					"implied_vol": p.implied_vol,
					"option_type": p.type,
			})()
			)

		builder = IVSurfaceBuilder()
		df_flat = builder.to_dataframe(contracts_like)
		call_surface, put_surface = IVSurfaceBuilder.build_iv_surfaces(df_flat)

		# Drop columns/rows that are entirely NaN
		if not call_surface.empty:
			call_surface = call_surface.dropna(axis=0, how="all").dropna(axis=1, how="all")
		if not put_surface.empty:
			put_surface = put_surface.dropna(axis=0, how="all").dropna(axis=1, how="all")

		if call_surface.empty and put_surface.empty:
			st.warning("IV call/put surfaces are empty after filtering NaNs.")
			return

		# 5) Show option chain tables (calls & puts)
		st.subheader("Option Chain (Filtered)")
		chain_df = chain.to_dataframe()
		chain_df = chain_df[chain_df["maturity"] > 0].copy()
		if option_type_filter == "Calls":
			chain_df = chain_df[chain_df["type"] == "C"]
		elif option_type_filter == "Puts":
			chain_df = chain_df[chain_df["type"] == "P"]

		calls_table = chain_df[chain_df["type"] == "C"]
		puts_table = chain_df[chain_df["type"] == "P"]

		col_calls, col_puts = st.columns(2)
		with col_calls:
			st.markdown("**Calls**")
			if not calls_table.empty:
				st.dataframe(calls_table)
			else:
				st.info("No call contracts after filters.")

		with col_puts:
			st.markdown("**Puts**")
			if not puts_table.empty:
				st.dataframe(puts_table)
			else:
				st.info("No put contracts after filters.")

		st.markdown("---")

		# 6) Plot IV surfaces
		col1, col2 = st.columns(2)
		with col1:
			st.subheader("Call IV Surface")
			if not call_surface.empty:
				fig_calls = plot_iv_surface(call_surface)
				st.plotly_chart(fig_calls, use_container_width=True)
			else:
				st.info("No call options available for surface.")

		with col2:
			st.subheader("Put IV Surface")
			if not put_surface.empty:
				fig_puts = plot_iv_surface(put_surface)
				st.plotly_chart(fig_puts, use_container_width=True)
			else:
				st.info("No put options available for surface.")

		st.subheader("Chain summary")
		st.write(f"Underlying: **{ticker}**, spot: **{spot:.2f}**")
		st.write(f"Contracts used (with IV): **{len(points)}**")


if __name__ == "__main__":
	main()

