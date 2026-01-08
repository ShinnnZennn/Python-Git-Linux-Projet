import streamlit as st
import pandas as pd
import numpy as np

from single_asset.get_data_yahoo import get_btc_data
from single_asset.strategies import (
    compute_buy_and_hold_curve,
    compute_momentum_sma_strategy,
)
from single_asset.metrics import (
    compute_total_return,
    compute_annualized_return,
    compute_annualized_volatility,
    compute_sharpe_ratio,
    compute_max_drawdown
)

from single_asset.predictive_models import ridge_regression_forecast


def run_quant_a():
    # ============================================================
    # CONFIG
    # ============================================================
    st.set_page_config(
        page_title="Quant A - BTC Single Asset",
        layout="wide"
    )

    st.title("Quant A — Single Asset : Bitcoin (BTC-USD)")


    # ============================================================
    # SIDEBAR PARAMETERS
    # ============================================================
    st.sidebar.header("Parameters")

    period = st.sidebar.selectbox(
        "Period",
        ["7d", "1mo", "2mo", "3mo", "6mo", "1y"],
        index=1  # "30d" default
    )

    interval = st.sidebar.selectbox(
        "Interval",
        ["1h", "4h", "1d"],
        index=0  # "1h" default
    )

    strategy_name = st.sidebar.selectbox(
        "Strategy",
        ["None (Price)", "Buy & Hold", "Momentum (SMA)"],
        index=1
    )


    # Parameters for strategies
    if strategy_name == "Momentum (SMA)":
        st.sidebar.subheader("Parameters Momentum")
        fast_window = st.sidebar.slider(
            "Fast Window (short SMA)",
            min_value=3,
            max_value=50,
            value=10,
            step=1
        )
        slow_window = st.sidebar.slider(
            "Slow Window (long SMA)",
            min_value=10,
            max_value=200,
            value=50,
            step=1
        )
    else:
        fast_window = 10
        slow_window = 50


    initial_capital = st.sidebar.slider(
        "Initial Capital (USD)",
        min_value=10_000,
        max_value=200_000,
        value=50_000,
        step=5_000
    )

    # ============================================================
    # FORECAST
    # ============================================================

    st.sidebar.subheader("Forecast")

    enable_forecast = st.sidebar.checkbox(
        "Enable price forecast",
        value=False
    )

    forecast_horizon = st.sidebar.slider(
        "Forecast horizon (periods)",
        min_value=30,
        max_value=300,
        value=100,
        step=10
    )

    st.sidebar.markdown(
        "**Ticker :** BTC-USD<br>**Source :** Yahoo Finance",
        unsafe_allow_html=True
    )

    # ============================================================
    # DATA GATHERING
    # ============================================================
    @st.cache_data(show_spinner=True)
    def load_data(period: str, interval: str):
        df = get_btc_data(period=period, interval=interval)
        df = df.copy()
        df["close"] = df["close"].astype(float)
        return df


    try:
        df = load_data(period, interval)
    except Exception as e:
        st.error(f"Error while loading data : {e}")
        st.stop()

    if df.empty:
        st.warning("No data available.")
        st.stop()


    # ============================================================
    # COMMPUTE
    # ============================================================
    last_close = float(df["close"].iloc[-1])
    last_ts = df["timestamp"].iloc[-1]

    if len(df) > 1:
        prev_close = float(df["close"].iloc[-2])
        delta = last_close - prev_close
        pct = (delta / prev_close * 100.0) if prev_close != 0 else 0.0
    else:
        prev_close = None
        delta = 0.0
        pct = 0.0

    current_price = last_close

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Current Price (close)",
        f"{current_price:,.2f} USD",
        f"{delta:+.2f} USD"
    )

    col2.metric(
        "Variation %",
        f"{pct:+.2f} %",
    )

    col3.write(f"**Last update :** {last_ts}")


    # ============================================================
    # PRINCIPAL PLOT
    # ============================================================
    st.subheader("Bitcoin price and strategy")

    df_plot = df.copy()
    df_plot = df_plot.set_index("timestamp")

    chart_df = pd.DataFrame(index=df_plot.index)
    chart_df["Price (USD)"] = df_plot["close"]

    if strategy_name == "Buy & Hold":
        strat_df = compute_buy_and_hold_curve(
            df,
            initial_capital=initial_capital
        )
        strat_df = strat_df.set_index("timestamp")
        chart_df["Buy & Hold (equity)"] = strat_df["equity_curve"]

    elif strategy_name == "Momentum (SMA)":
        strat_df = compute_momentum_sma_strategy(
            df,
            fast_window,
            slow_window,
            initial_capital=initial_capital
        )
        strat_df = strat_df.set_index("timestamp")
        chart_df["Momentum SMA"] = strat_df["equity_curve"]

    # ============================================================
    # FORECAST
    # ============================================================

    if enable_forecast:
        forecast_df = ridge_regression_forecast(
            df,
            horizon=forecast_horizon
        ).set_index("timestamp")


        # 1️⃣ Extend index FIRST
        full_index = chart_df.index.union(forecast_df.index)
        chart_df = chart_df.reindex(full_index)

        # 2️⃣ Create forecast columns
        chart_df["Forecast"] = np.nan
        chart_df["Lower CI"] = np.nan
        chart_df["Upper CI"] = np.nan

        # 3️⃣ Assign forecast values (NOW SAFE)
        chart_df.loc[forecast_df.index, "Forecast"] = forecast_df["prediction"]
        chart_df.loc[forecast_df.index, "Lower CI"] = forecast_df["lower_ci"]
        chart_df.loc[forecast_df.index, "Upper CI"] = forecast_df["upper_ci"]




    # ============================================================
    # PLOT
    # ============================================================

    plot_cols = ["Price (USD)"]
    # Always show strategy
    if strategy_name == "Buy & Hold":
        plot_cols.append("Buy & Hold (equity)")
    elif strategy_name == "Momentum (SMA)":
        plot_cols.append("Momentum SMA")

    # Add forecast if enabled
    if enable_forecast:
        plot_cols += ["Forecast", "Lower CI", "Upper CI"]

    st.line_chart(chart_df[plot_cols])



    # ============================================================
    # METRICS
    # ============================================================
    if strategy_name == "None (Price)":
        equity = chart_df["Price (USD)"]
    else:
        equity = strat_df["equity_curve"]


    if interval == "1h":
        periods_per_year = 24 * 365 # 8760
    elif interval == "4h":
        periods_per_year = 6 * 365 # 2190
    else:  # "1d"
        periods_per_year = 365

    long_enough_periods = ["1mo", "2mo", "3mo", "6mo", "1y"]

    if period in long_enough_periods:
        annualize = True
    else:
        annualize = False
        st.warning("The period selected is too short to compute annualized metrics reliably.")


    total_ret = compute_total_return(equity)

    if annualize:
        ann_ret = compute_annualized_return(equity, periods_per_year)
        ann_vol = compute_annualized_volatility(equity, periods_per_year)
        sharpe = compute_sharpe_ratio(equity, periods_per_year)
    else:
        ann_ret = float("nan")
        ann_vol = float("nan")
        sharpe = float("nan")

    mdd = compute_max_drawdown(equity)


    st.subheader("Performance Metrics")

    m1, m2, m3, m4, m5 = st.columns(5)

    m1.metric("Total Return", f"{total_ret*100:.2f} %")
    m2.metric("Annualized Return", f"{ann_ret*100:.2f} %")
    m3.metric("Annualized Volatility", f"{ann_vol*100:.2f} %")
    m4.metric("Sharpe Ratio", f"{sharpe:.2f}")
    m5.metric("Max Drawdown", f"{mdd*100:.2f} %")


    with st.expander("Show raw data"):
        st.dataframe(df.tail(50))
