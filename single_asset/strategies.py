import pandas as pd


def compute_buy_and_hold_curve(
    df: pd.DataFrame,
    initial_capital: float = 100000.0,   # capital USD
    buy_fee: float = 0.001,            # 0.1% fees
    daily_fee: float = 0.0001          # 0.01% fees per day
) -> pd.DataFrame:
    """
    Buy & Hold Strategy in USD with fees:
    """
    prices = df["close"].astype(float).squeeze()


    # Timestamp 
    timestamps = pd.to_datetime(df["timestamp"])

    #  Buy fees at the beginning
    capital_after_fee = initial_capital * (1 - buy_fee)

    # Number of BTC purchased
    first_price = prices.iloc[0]
    btc_qty = capital_after_fee / first_price

    # Daily costs calculation
    days = (timestamps - timestamps.iloc[0]).dt.days
    holding_decay = (1 - daily_fee) ** days

    # Portfolio value over time
    equity_curve = btc_qty * prices * holding_decay

    # Dataframe output
    out = pd.DataFrame()
    out["timestamp"] = timestamps
    out["equity_curve"] = equity_curve 
    out["returns"] = out["equity_curve"].pct_change().fillna(0.0)

    return out



def compute_momentum_sma_strategy(
    df: pd.DataFrame,
    fast_window: int = 10,
    slow_window: int = 50,
    initial_capital: float = 100000.0
) -> pd.DataFrame:
    """
    Momentum SMA in USD :
    - We invest `initial_capital`
    - When SMA_fast > SMA_slow : full BTC
    - Otherwise : full cash
    - Tracking the portfolio in dollars
    """

    prices = df["close"].astype(float).squeeze()
    timestamps = pd.to_datetime(df["timestamp"])

    # SMA
    sma_fast = prices.rolling(fast_window, min_periods=fast_window).mean()
    sma_slow = prices.rolling(slow_window, min_periods=slow_window).mean()

    # Position : 1 = long BTC, 0 = cash
    position = (sma_fast > sma_slow).astype(float)

    # Returns
    price_returns = prices.pct_change().fillna(0.0)

    # Strategy returns
    strategy_returns = position.shift(1).fillna(0.0) * price_returns

    # Equity en USD
    equity_curve = initial_capital * (1 + strategy_returns).cumprod()

    out = pd.DataFrame({
        "timestamp": timestamps,
        "position": position,
        "returns": strategy_returns,
        "equity_curve": equity_curve,
    })

    return out
