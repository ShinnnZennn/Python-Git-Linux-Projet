import streamlit as st
from single_asset.app_single_asset import run_quant_a

st.set_page_config(page_title="Linux Git Python Project", layout="wide")

page = st.sidebar.selectbox(
    "Module",
    ["Single Asset (Quant A)"]
)

if page == "Single Asset (Quant A)":
    run_quant_a()
