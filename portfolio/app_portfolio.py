import streamlit as st
import numpy as np
import pandas as pd

from portfolio.get_data_multi import get_multi_asset_data
from portfolio.portfolio_strategies import (
    compute_asset_returns,
    compute_portfolio_equity,
    compute_portfolio_stats,
)


def run_portfolio_app():
    # ============================================================
    # PAGE CONFIG
    # ============================================================
    st.set_page_config(
        page_title="Quant B - Portfolio",
        layout="wide"
    )

    st.title("Quant B — Multi-Asset Portfolio Analysis")

    # ============================================================
    # SIDEBAR — PARAMETERS
    # ============================================================
    st.sidebar.header("Portfolio Parameters")

    tickers_input = st.sidebar.text_input(
        "Tickers (comma separated)",
        value="AAPL,MSFT,TSLA"
    )

    tickers = [t.strip().upper() for t in tickers_input.split(",") if t.strip()]

    period = st.sidebar.selectbox(
        "Period",
        ["1mo", "3mo", "6mo", "1y", "2y"],
        index=2
    )

    interval = st.sidebar.selectbox(
        "Interval",
        ["1h", "1d"],
        index=1
    )

    rebalancing = st.sidebar.selectbox(
        "Rebalancing Frequency",
        ["none", "weekly", "monthly"],
        index=0
    )

    initial_capital = st.sidebar.slider(
        "Initial Capital (USD)",
        min_value=10_000,
        max_value=500_000,
        value=100_000,
        step=10_000
    )

    st.sidebar.markdown("---")
    st.sidebar.markdown("**Source:** Yahoo Finance")

    # ============================================================
    # DATA LOADING
    # ============================================================
    @st.cache_data(show_spinner=True)
    def load_data(tickers, period, interval):
        return get_multi_asset_data(
            tickers=tickers,
            period=period,
            interval=interval
        )

    try:
        prices_df = load_data(tickers, period, interval)
    except Exception as e:
        st.error(f"Error while loading data: {e}")
        st.stop()

    if prices_df.empty:
        st.warning("No data available.")
        st.stop()

    # ============================================================
    # RETURNS
    # ============================================================
    returns_df = compute_asset_returns(prices_df)

    asset_cols = [c for c in returns_df.columns if c != "timestamp"]
    n_assets = len(asset_cols)


    # ============================================================
    # WEIGHTS
    # ============================================================
    st.sidebar.subheader("Portfolio Weights")

    weights = []
    for asset in asset_cols:
        w = st.sidebar.slider(
            f"Weight {asset}",
            min_value=0.0,
            max_value=1.0,
            value=1.0 / n_assets,
            step=0.05
        )
        weights.append(w)

    weights = np.array(weights)

    if weights.sum() == 0:
        st.error("Sum of weights must be greater than 0.")
        st.stop()

    weights = weights / weights.sum()

    # ============================================================
    # PORTFOLIO SIMULATION
    # ============================================================
    portfolio_df = compute_portfolio_equity(
        returns_df=returns_df,
        weights=weights,
        initial_capital=initial_capital,
        rebalancing=rebalancing,
    )

    # ============================================================
    # ASSET VALUE CURVES
    # ============================================================

    returns_only = returns_df.set_index("timestamp")
    asset_cols = [c for c in returns_only.columns]

    # Initial USD allocation per asset
    initial_allocations = weights * initial_capital

    asset_value_df = pd.DataFrame(index=returns_only.index)

    for i, asset in enumerate(asset_cols):
        asset_value_df[asset] = (
            initial_allocations[i] * (1 + returns_only[asset]).cumprod()
        )


    # Add portfolio total value
    portfolio_series = (
        portfolio_df
        .set_index("timestamp")["portfolio_value"]
    )

    asset_value_df["Portfolio"] = portfolio_series

    # ============================================================
    # MAIN VISUAL COMPARISON
    # ============================================================

    st.subheader("Asset Allocations vs Portfolio Value (USD)")
    st.line_chart(asset_value_df)

    # ============================================================
    # METRICS
    # ============================================================
    periods_per_year = 252 if interval == "1d" else 24 * 365

    stats = compute_portfolio_stats(
        returns_df=returns_df,
        weights=weights,
        periods_per_year=periods_per_year
    )

    st.subheader("Portfolio Performance Metrics")

    c1, c2, c3 = st.columns(3)

    c1.metric(
        "Annualized Return",
        f"{stats['portfolio_mean_return'] * 100:.2f} %"
    )

    c2.metric(
        "Annualized Volatility",
        f"{stats['portfolio_volatility'] * 100:.2f} %"
    )

    c3.metric(
        "Sharpe Ratio",
        f"{stats['portfolio_sharpe_ratio']:.2f}"
    )

    # ============================================================
    # ASSET LEVEL METRICS
    # ============================================================
    st.subheader("Asset-Level Statistics")

    asset_stats_df = pd.DataFrame({
        "Annualized Return": stats["asset_mean_returns"],
        "Annualized Volatility": stats["asset_volatility"],
        "Weight": weights,
    })

    st.dataframe(asset_stats_df.style.format({
        "Annualized Return": "{:.2%}",
        "Annualized Volatility": "{:.2%}",
        "Weight": "{:.2%}",
    }))

    # ============================================================
    # CORRELATION MATRIX
    # ============================================================
    st.subheader("Asset Correlation Matrix")

    corr_df = stats["correlation_matrix"]

    st.dataframe(
        corr_df.round(2)
    )


    # ============================================================
    # DIVERSIFICATION EFFECT
    # ============================================================

    weighted_avg_vol = float(
        np.sum(weights * stats["asset_volatility"].values)
    )

    diversification_gain = weighted_avg_vol - stats["portfolio_volatility"]

    st.subheader("Diversification Effect")

    d1, d2 = st.columns(2)

    d1.metric(
        "Weighted Avg Asset Volatility",
        f"{weighted_avg_vol * 100:.2f} %"
    )

    d2.metric(
        "Diversification Gain",
        f"{diversification_gain * 100:.2f} %",
        help="Difference between weighted average volatility and portfolio volatility"
    )


    # ============================================================
    # RAW DATA
    # ============================================================
    with st.expander("Show raw data"):
        st.write("Prices")
        st.dataframe(prices_df.tail(50))
        st.write("Returns")
        st.dataframe(returns_df.tail(50))


# ============================================================
# ENTRY POINT
# ============================================================
if __name__ == "__main__":
    run_portfolio_app()
