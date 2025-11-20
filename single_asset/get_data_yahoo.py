import yfinance as yf
import pandas as pd

TICKER = "BTC-USD"


def get_btc_data(
    period: str = "30d",
    interval: str = "1h"
) -> pd.DataFrame:
    """
    Récupère les données BTC-USD depuis Yahoo Finance et renvoie
    un DataFrame propre contenant :
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
        raise RuntimeError("Aucune donnée récupérée depuis Yahoo Finance.")

    df = df.reset_index()

    # Selon le cas, Yahoo renvoie 'Date' ou 'Datetime'
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


if __name__ == "__main__":
    print("Téléchargement des données BTC-USD...")
    df = get_btc_data(period="30d", interval="1h")

    print("\n--- 10 premières lignes ---")
    print(df.head(10))

    print("\n--- 10 dernières lignes ---")
    print(df.tail(10))

    print("\n--- Prix de clôture récents ---")
    print(df[['timestamp', 'close']].tail(10))
