import yfinance as yf
import pandas as pd

TICKER = "BTC-USD"


def get_btc_data(
    period: str = "30d",
    interval: str = "1h"
) -> pd.DataFrame:
    """
    Gathering BTC-USD data from Yahoo Finance and returning
    a clean DataFrame containing:
    timestamp, open, high, low, close, volume
    """

    df = yf.download(
        TICKER,
        period=period,
        interval=interval,
        auto_adjust=False,
        progress=False
    )

    if df.empty:
        raise RuntimeError("No available data from Yahoo Finance.")

    df = df.reset_index()

    # 'Date' or 'Datetime'
    if "Datetime" in df.columns:
        time_col = "Datetime"
    else:
        time_col = "Date"

    df = df.rename(
        columns={
            time_col: "timestamp",
            "Open": "open",
            "High": "high",
            "Low": "low",
            "Close": "close",
            "Volume": "volume",
        }
    )

    df = df[["timestamp", "open", "high", "low", "close", "volume"]]

    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.dropna().sort_values("timestamp").reset_index(drop=True)

    return df