import streamlit as st
import pandas as pd

from .loader import load_asset
from .strategies import buy_and_hold, ma_crossover
from .metrics import max_drawdown, annualized_vol, sharpe_ratio

def run():
    st.header("Quant A – Single Asset Analysis")

    ticker = st.text_input("Ticker (ex: AAPL, MSFT, ^FCHI)", "AAPL")
    period = st.selectbox("Période", ["1mo", "3mo", "6mo", "1y", "5y"], index=3)
    interval = st.selectbox("Intervalle", ["1d", "1h", "30m", "15m"], index=0)

    strategy_name = st.selectbox("Stratégie", ["Buy & Hold", "MA Crossover"])
    short_window = st.slider("MA courte", 5, 50, 20)
    long_window = st.slider("MA longue", 20, 200, 50)

    if st.button("Analyser"):
        df = load_asset(ticker, period=period, interval=interval)
        price = df["Close"]

        buyhold_cum = buy_and_hold(price)

        if strategy_name == "Buy & Hold":
            strat_cum = buyhold_cum
        else:
            strat_cum = ma_crossover(price, short_window=short_window, long_window=long_window)

        st.line_chart(
            pd.DataFrame({
                "Price": price.values,
                "Strategy": strat_cum.values,
            }, index=df["Date"])
        )

        returns = price.pct_change().fillna(0)
        strat_returns = strat_cum.pct_change().fillna(0)

        col1, col2, col3 = st.columns(3)
        col1.metric("Max Drawdown", f"{max_drawdown(strat_cum):.2%}")
        col2.metric("Vol annualisée", f"{annualized_vol(strat_returns):.2%}")
        col3.metric("Sharpe Ratio", f"{sharpe_ratio(strat_returns):.2f}")
