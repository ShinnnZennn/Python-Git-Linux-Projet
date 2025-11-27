import pandas as pd


def compute_buy_and_hold_curve(
    df: pd.DataFrame,
    initial_capital: float = 1.0
) -> pd.DataFrame:
    """
    Stratégie Buy & Hold :
    - On achète au début
    - On garde jusqu'à la fin
    - L'equity curve suit le prix normalisé (base initial_capital)

    Retourne un DataFrame avec :
        timestamp, equity_curve, returns
    """

    prices = df["close"].astype(float)

    equity_curve = initial_capital * (prices / prices.iloc[0])

    out = pd.DataFrame()
    out["timestamp"] = pd.to_datetime(df["timestamp"])
    out["equity_curve"] = equity_curve
    out["returns"] = out["equity_curve"].pct_change().fillna(0.0)

    return out


def compute_momentum_sma_strategy(
    df: pd.DataFrame,
    fast_window: int = 10,
    slow_window: int = 50,
    initial_capital: float = 1.0
) -> pd.DataFrame:
    """
    Stratégie Momentum basée sur deux moyennes mobiles (SMA crossover) :

    - SMA rapide (fast_window)
    - SMA lente (slow_window)
    - Si SMA rapide > SMA lente -> position = 1 (long)
      Sinon -> position = 0 (cash)

    Les rendements de la stratégie sont :
        position(t-1) * retour_pct_du_prix(t)

    Retourne un DataFrame avec :
        timestamp, equity_curve, position, returns
    """

    prices = df["close"].astype(float)

    sma_fast = prices.rolling(window=fast_window, min_periods=fast_window).mean()
    sma_slow = prices.rolling(window=slow_window, min_periods=slow_window).mean()

    # Position long uniquement quand la rapide est au-dessus de la lente
    position = (sma_fast > sma_slow).astype(float)

    # Rendements du prix
    price_returns = prices.pct_change().fillna(0.0)

    # On applique la position avec un décalage d'un pas (on réagit au signal du jour d'avant)
    strategy_returns = position.shift(1).fillna(0.0) * price_returns

    equity_curve = initial_capital * (1 + strategy_returns).cumprod()

    out = pd.DataFrame()
    out["timestamp"] = pd.to_datetime(df["timestamp"])
    out["position"] = position
    out["returns"] = strategy_returns
    out["equity_curve"] = equity_curve

    return out
