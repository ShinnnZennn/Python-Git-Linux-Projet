import pandas as pd
import numpy as np


# ============================================================
# DATA PREPARATION
# ============================================================

def extract_close_prices(prices_df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract close prices for each asset from a multi-asset price DataFrame.

    Input:
        prices_df:
            - column "timestamp"
            - other columns are a MultiIndex: (ticker, field)
              where field ∈ {open, high, low, close, volume}

    Output:
        close_df:
            - index: timestamp
            - columns: one column per ticker (AAPL, MSFT, BTC-USD, ...)
    """
    df = prices_df.copy()
    df = df.set_index("timestamp")

    # Multi-asset case (MultiIndex columns)
    if isinstance(df.columns, pd.MultiIndex):
        close_cols = [col for col in df.columns if col[1] == "close"]
        close_df = df[close_cols].copy()
        close_df.columns = [col[0] for col in close_df.columns]

    # Single-asset fallback
    else:
        close_df = df[["close"]].copy()

    close_df = close_df.astype(float)
    close_df.index = pd.to_datetime(close_df.index)
    close_df = close_df.sort_index()

    return close_df


def compute_asset_returns(prices_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute simple returns for each asset using close prices.

    Formula:
        r_t = close_t / close_{t-1} - 1

    Output:
        returns_df:
            - column "timestamp"
            - one column per asset
    """
    close_df = extract_close_prices(prices_df)

    returns = close_df.pct_change().dropna(how="all")

    returns = returns.reset_index()
    returns = returns.rename(columns={"index": "timestamp"})

    return returns


# ============================================================
# PORTFOLIO SIMULATION
# ============================================================

def compute_portfolio_equity(
    returns_df: pd.DataFrame,
    weights,
    initial_capital: float = 1.0,
    rebalancing: str = "none",
) -> pd.DataFrame:
    """
    Simulate a multi-asset portfolio over time.

    Parameters:
        returns_df:
            - column "timestamp"
            - other columns: asset returns
        weights:
            - list or array of portfolio weights (must match asset columns)
        initial_capital:
            - starting portfolio value (e.g. 1.0 or 100000)
        rebalancing:
            - "none"    : Buy & Hold
            - "weekly"  : weekly rebalancing
            - "monthly" : monthly rebalancing

    Output:
        DataFrame with:
            - timestamp
            - portfolio_value
            - portfolio_returns
    """
    if "timestamp" not in returns_df.columns:
        raise ValueError("returns_df must contain a 'timestamp' column.")

    df = returns_df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp")

    asset_cols = [c for c in df.columns if c != "timestamp"]
    if len(asset_cols) == 0:
        raise ValueError("returns_df must contain at least one asset column.")

    # Normalize weights
    weights = np.asarray(weights, dtype=float)
    if len(weights) != len(asset_cols):
        raise ValueError("Number of weights must match number of assets.")
    if weights.sum() == 0:
        raise ValueError("Sum of weights must be > 0.")
    weights = weights / weights.sum()

    df = df.set_index("timestamp")
    returns = df[asset_cols].astype(float).dropna(how="all")

    dates = returns.index
    returns_matrix = returns.values

    # Rebalancing schedule
    if rebalancing == "none":
        rebalance_dates = set()

    elif rebalancing == "weekly":
        rebalance_dates = set(
            pd.Series(dates)
            .groupby(pd.Series(dates).dt.to_period("W"))
            .first()
            .values
        )

    elif rebalancing == "monthly":
        rebalance_dates = set(
            pd.Series(dates)
            .groupby(pd.Series(dates).dt.to_period("M"))
            .first()
            .values
        )

    else:
        raise ValueError("rebalancing must be 'none', 'weekly' or 'monthly'.")


    portfolio_value = initial_capital
    current_weights = weights.copy()

    timestamps = []
    portfolio_values = []
    portfolio_returns = []

    for i, (ts, r_vec) in enumerate(zip(dates, returns_matrix)):
        if i == 0:
            timestamps.append(ts)
            portfolio_values.append(portfolio_value)
            portfolio_returns.append(0.0)
            continue

        r_p = float(np.dot(current_weights, r_vec))
        portfolio_value *= (1.0 + r_p)

        timestamps.append(ts)
        portfolio_values.append(portfolio_value)
        portfolio_returns.append(r_p)

        # Weight drift
        asset_values = current_weights * (1.0 + r_vec)
        if asset_values.sum() > 0:
            current_weights = asset_values / asset_values.sum()

        # Rebalancing
        if ts in rebalance_dates:
            current_weights = weights.copy()

    return pd.DataFrame({
        "timestamp": timestamps,
        "portfolio_value": portfolio_values,
        "portfolio_returns": portfolio_returns,
    })


# ============================================================
# PORTFOLIO STATISTICS
# ============================================================

def compute_portfolio_stats(
    returns_df: pd.DataFrame,
    weights,
    periods_per_year: int = 252,
) -> dict:
    """
    Compute annualized statistics for a multi-asset portfolio.

    Returns:
        dict with:
            - asset_mean_returns
            - asset_volatility
            - covariance_matrix
            - correlation_matrix
            - portfolio_mean_return
            - portfolio_volatility
            - portfolio_sharpe_ratio
    """
    if "timestamp" not in returns_df.columns:
        raise ValueError("returns_df must contain a 'timestamp' column.")

    df = returns_df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp")

    asset_cols = [c for c in df.columns if c != "timestamp"]
    if len(asset_cols) == 0:
        raise ValueError("returns_df must contain asset return columns.")

    df = df.set_index("timestamp")
    asset_returns = df[asset_cols].astype(float).dropna(how="all")

    weights = np.asarray(weights, dtype=float)
    weights = weights / weights.sum()

    mean_returns = asset_returns.mean() * periods_per_year
    vol_returns = asset_returns.std() * np.sqrt(periods_per_year)

    cov_matrix = asset_returns.cov() * periods_per_year
    corr_matrix = asset_returns.corr()

    portfolio_return = float(np.dot(weights, mean_returns.values))
    portfolio_volatility = float(
        np.sqrt(np.dot(weights.T, np.dot(cov_matrix.values, weights)))
    )

    sharpe_ratio = (
        portfolio_return / portfolio_volatility
        if portfolio_volatility > 0 else np.nan
    )

    return {
        "asset_mean_returns": mean_returns,
        "asset_volatility": vol_returns,
        "covariance_matrix": cov_matrix,
        "correlation_matrix": corr_matrix,
        "portfolio_mean_return": portfolio_return,
        "portfolio_volatility": portfolio_volatility,
        "portfolio_sharpe_ratio": sharpe_ratio,
    }
