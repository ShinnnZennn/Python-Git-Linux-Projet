import streamlit as st
from single_asset.ui import run as run_single_asset

st.set_page_config(page_title="Linux Git Python Project", layout="wide")

page = st.sidebar.selectbox("Module", ["Single Asset (Quant A)"])

if page == "Single Asset (Quant A)":
    run_single_asset()
