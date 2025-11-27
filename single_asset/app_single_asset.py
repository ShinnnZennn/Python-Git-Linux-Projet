import streamlit as st
import pandas as pd

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



def run_quant_a():
    # --- Configuration de la page ---
    st.set_page_config(
        page_title="Quant A - BTC Single Asset",
        layout="wide"
    )

    st.title("Quant A — Analyse d'un seul actif : Bitcoin (BTC-USD)")


    # --- Barre latérale : paramètres ---
    st.sidebar.header("Paramètres des données")

    period = st.sidebar.selectbox(
        "Période",
        ["7d", "30d", "60d", "1y"],
        index=1  # "30d" par défaut
    )

    interval = st.sidebar.selectbox(
        "Intervalle",
        ["1h", "4h", "1d"],
        index=0  # "1h" par défaut
    )

    strategy_name = st.sidebar.selectbox(
        "Stratégie",
        ["Aucune (prix seul)", "Buy & Hold", "Momentum (SMA)"],
        index=1
    )

    st.sidebar.markdown(
        "**Ticker :** BTC-USD  \n**Source :** Yahoo Finance"
    )

    # Paramètres de la stratégie momentum
    if strategy_name == "Momentum (SMA)":
        st.sidebar.subheader("Paramètres Momentum")
        fast_window = st.sidebar.slider(
            "Fenêtre rapide (SMA courte)",
            min_value=3,
            max_value=50,
            value=10,
            step=1
        )
        slow_window = st.sidebar.slider(
            "Fenêtre lente (SMA longue)",
            min_value=10,
            max_value=200,
            value=50,
            step=1
        )
    else:
        # valeurs par défaut (non utilisées si pas momentum)
        fast_window = 10
        slow_window = 50


    # --- Récupération des données ---
    @st.cache_data(show_spinner=True)
    def load_data(period: str, interval: str):
        df = get_btc_data(period=period, interval=interval)
        df = df.copy()
        df["close"] = df["close"].astype(float)
        return df


    try:
        df = load_data(period, interval)
    except Exception as e:
        st.error(f"Erreur lors du téléchargement des données : {e}")
        st.stop()

    if df.empty:
        st.warning("Aucune donnée disponible.")
        st.stop()


    # --- Calculs pour l'affichage des métriques ---
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


    # --- métriques en haut ---
    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Prix actuel (close)",
        f"{current_price:,.2f} USD",
        f"{delta:+.2f} USD"
    )

    col2.metric(
        "Variation %",
        f"{pct:+.2f} %",
    )

    col3.write(f"**Dernière mise à jour :** {last_ts}")


    # --- Graphique principal : prix + stratégie ---
    st.subheader("Historique de Bitcoin et stratégie (indice base 1.0)")

    df_plot = df.copy()
    df_plot = df_plot.set_index("timestamp")

    chart_df = pd.DataFrame(index=df_plot.index)
    chart_df["Prix (normalisé)"] = df_plot["close"] / df_plot["close"].iloc[0]

    if strategy_name == "Buy & Hold":
        strat_df = compute_buy_and_hold_curve(df)
        strat_df = strat_df.set_index("timestamp")
        chart_df["Stratégie Buy & Hold"] = strat_df["equity_curve"]

    elif strategy_name == "Momentum (SMA)":
        strat_df = compute_momentum_sma_strategy(
            df,
            fast_window=fast_window,
            slow_window=slow_window,
            initial_capital=1.0,
        )
        strat_df = strat_df.set_index("timestamp")
        chart_df["Stratégie Momentum (SMA)"] = strat_df["equity_curve"]

    st.line_chart(chart_df, x_label="Date", y_label="Indice (base 1.0)")

    # METRICS QUANT A
    equity = strat_df["equity_curve"] if strategy_name != "Aucune (prix seul)" else chart_df["Prix (normalisé)"]

    # Rendements logarithmiques ou percent change
    returns = equity.pct_change().fillna(0.0)

    # Détermine le nombre de périodes par an selon l'interval
    if interval == "1h":
        periods_per_year = 24 * 365
    elif interval == "4h":
        periods_per_year = 6 * 365
    else:  # "1d"
        periods_per_year = 365

    total_ret = compute_total_return(equity)
    ann_ret = compute_annualized_return(equity, periods_per_year)
    ann_vol = compute_annualized_volatility(returns, periods_per_year)
    sharpe = compute_sharpe_ratio(returns, periods_per_year)
    mdd = compute_max_drawdown(equity)


    st.subheader("Metrics de performance")

    m1, m2, m3, m4, m5 = st.columns(5)

    m1.metric("Return total", f"{total_ret*100:.2f} %")
    m2.metric("Return annualisé", f"{ann_ret*100:.2f} %")
    m3.metric("Volatilité annualisée", f"{ann_vol*100:.2f} %")
    m4.metric("Sharpe Ratio", f"{sharpe:.2f}")
    m5.metric("Max Drawdown", f"{mdd*100:.2f} %")

    # --- Tableau de données (optionnel) ---
    with st.expander("Voir les données brutes"):
        st.dataframe(df.tail(50))
