import yfinance as yf
import pandas as pd


def get_multi_asset_data(
    tickers: list,
    period: str = "30d",
    interval: str = "1h"
) -> pd.DataFrame:
    """
    Download historical price data from Yahoo Finance for multiple assets.
    """

    if not tickers or len(tickers) < 2:
        raise ValueError("Please provide at least two tickers.")

    # Download data in a single request
    df = yf.download(
        tickers=tickers,
        period=period,
        interval=interval,
        auto_adjust=False,
        progress=False,
        group_by="ticker"
    )

    if df.empty:
        raise RuntimeError("No data returned from Yahoo Finance.")

    # Normalize the data structure
    final_frames = []

    for ticker in tickers:
        try:
            sub_df = df[ticker].copy()
        except Exception:
            # Missing ticker data
            sub_df = pd.DataFrame()

        if sub_df.empty:
            print(f"[WARNING] No data received for {ticker}.")
            continue

        # Standardize column names
        sub_df = sub_df.rename(columns={
            "Open": "open",
            "High": "high",
            "Low": "low",
            "Close": "close",
            "Volume": "volume",
        })

        # Add a column level with the ticker name
        sub_df.columns = pd.MultiIndex.from_product(
            [[ticker], sub_df.columns]
        )

        final_frames.append(sub_df)

    # Merge all assets on the timestamp index
    full_df = pd.concat(final_frames, axis=1)

    # Clean and sort
    full_df = full_df.dropna(how="all")
    full_df = full_df.sort_index()

    # Reset index to have timestamp as a column
    full_df = full_df.reset_index()
    full_df = full_df.rename(columns={"Datetime": "timestamp", "Date": "timestamp"})
    full_df["timestamp"] = pd.to_datetime(full_df["timestamp"])

    return full_df


if __name__ == "__main__":
    TICKERS = ["AAPL", "MSFT", "TSLA"]

    print("Downloading multi-asset data...")
    df = get_multi_asset_data(TICKERS, period="30d", interval="1h")
    print(df.head())
