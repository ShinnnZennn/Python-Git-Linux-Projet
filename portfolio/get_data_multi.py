
import yfinance as yf
import pandas as pd


def get_multi_asset_data(
    tickers: list,   # mettre une liste d'actifs
    period: str = "30d",
    interval: str = "1h"
) -> pd.DataFrame:
    """
    Récupère les données Yahoo Finance pour plusieurs tickers.
    
    Retourne un DataFrame avec un MultiIndex colonnes :
        (ticker, open/close/high/low/volume)
    et l'index = timestamp commun.

    Exemple colonnes :
        AAPL   open
        AAPL   close
        TSLA   open
        TSLA   close
        ...

    Les données sont automatiquement alignées sur les timestamps communs.
    """

    if not tickers or len(tickers) < 2:
        raise ValueError("Veuillez fournir au moins 2 tickers.")

    # Téléchargement en une fois (beaucoup plus rapide)
    df = yf.download(
        tickers,
        period=period,
        interval=interval,
        auto_adjust=False,
        progress=False,
        group_by="ticker"  # important pour séparer chaque ticker
    )

    if df.empty:
        raise RuntimeError("Aucune donnée récupérée depuis Yahoo Finance.")

    # Quand un seul ticker : la structure n'est pas multi-index → on harmonise
    if len(tickers) == 1:
        df = {tickers[0]: df}
    else:
        # df est un MultiIndex (ticker, field)
        pass

    # --- Harmonisation : convertir en un DataFrame propre ---
    final_frames = []

    for ticker in tickers:
        try:
            sub_df = df[ticker].copy() # il y a qu'un seul ticker dans sub_df
        except Exception:
            # yfinance peut retourner un ticker manquant → on remplit NaN
            sub_df = pd.DataFrame()

        if sub_df.empty:
            print(f"[WARNING] Aucun data reçu pour {ticker}, colonnes remplies en NaN.")
            # on crée un DataFrame vide mais avec bon index si déjà existant
            continue

        sub_df = sub_df.rename(columns={
            "Open": "open",
            "High": "high",
            "Low": "low",
            "Close": "close",
            "Volume": "volume"
        })

        # On ajoute un niveau de colonne avec le ticker
        sub_df.columns = pd.MultiIndex.from_product([[ticker], sub_df.columns])  # c pour éviter d'avoir Open Open Open. La on aura Apple Open, BTC Open etc

        final_frames.append(sub_df)

    # Fusion par timestamp
    full_df = pd.concat(final_frames, axis=1) #on met tout ensemble

    # Nettoyage
    full_df = full_df.dropna(how="all")   # supprime lignes totalement vides
    full_df = full_df.sort_index()  #  On s’assure que les timestamps sont dans l’ordre croissant

    # Reset index → timestamp en colonne comme ton get_data_yahoo
    full_df = full_df.reset_index()
    full_df = full_df.rename(columns={"Datetime": "timestamp", "Date": "timestamp"})
    full_df["timestamp"] = pd.to_datetime(full_df["timestamp"])

    return full_df




if __name__ == "__main__":
    TICKERS = ["AAPL", "MSFT", "TSLA"]

    print("Téléchargement multi-actifs...")
    df = get_multi_asset_data(TICKERS, period="30d", interval="1h")
    print(df.head())

    print("\nCalcul des rendements...")
    r = compute_multi_returns(df)
    print(r.head())

