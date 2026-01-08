import streamlit as st
from streamlit_autorefresh import st_autorefresh

from single_asset.app_single_asset import run_quant_a
from portfolio.app_portfolio import run_portfolio_app

st.set_page_config(
    page_title="Linux Git Python Project",
    layout="wide"
)

# Auto refresh every 5 minutes
st_autorefresh(interval=5 * 60 * 1000, key="datarefresh")


# Module selection
page = st.sidebar.selectbox(
    "Module",
    [
        "Single Asset (Quant A)",
        "Portfolio (Quant B)",
    ]
)

# Routing
if page == "Single Asset (Quant A)":
    run_quant_a()

elif page == "Portfolio (Quant B)":
    run_portfolio_app()
