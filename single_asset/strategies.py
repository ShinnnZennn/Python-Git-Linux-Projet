import pandas as pd

def buy_and_hold(prices: pd.Series) -> pd.Series:
    returns = prices.pct_change().fillna(0)
    cum = (1 + returns).cumprod()
    return cum

def ma_crossover(prices: pd.Series, short_window: int = 20, long_window: int = 50) -> pd.Series:
    df = pd.DataFrame({"price": prices})
    df["ma_short"] = df["price"].rolling(short_window).mean()
    df["ma_long"] = df["price"].rolling(long_window).mean()
    df["position"] = (df["ma_short"] > df["ma_long"]).astype(int)
    df["ret"] = df["price"].pct_change().fillna(0)
    df["strategy_ret"] = df["position"].shift(1).fillna(0) * df["ret"]
    df["cum"] = (1 + df["strategy_ret"]).cumprod()
    return df["cum"]
