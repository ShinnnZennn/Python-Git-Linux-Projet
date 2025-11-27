import pandas as pd
import numpy as np


def extract_close_prices(prices_df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract close prices for each asset from the raw prices DataFrame.

    Input:
        prices_df:
            - column "timestamp"
            - other columns are a MultiIndex: (ticker, field)
              where field can be: open, high, low, close, volume

    Output:
        close_df:
            - index: timestamp
            - columns: one column per ticker (AAPL, MSFT, TSLA, ...)
    """
    df = prices_df.copy()
    df = df.set_index("timestamp")

    # Case 1: MultiIndex columns (for multiple tickers)
    if isinstance(df.columns, pd.MultiIndex):
        # We keep only the "close" prices for each ticker
        close_cols = [col for col in df.columns if col[1] == "close"]
        close_df = df[close_cols].copy()

        # Rename columns: keep only the ticker name
        close_df.columns = [col[0] for col in close_df.columns]

    # Case 2: Single asset with a simple "close" column
    else:
        close_df = df[["close"]].copy()

    close_df = close_df.astype(float)
    close_df.index = pd.to_datetime(close_df.index)
    close_df = close_df.sort_index()

    return close_df


def compute_asset_returns(prices_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute simple returns for each asset using close prices.

    Return formula:
        r_t = close_t / close_{t-1} - 1

    Input:
        prices_df: DataFrame from get_multi_asset_data()

    Output:
        returns_df:
            - column "timestamp"
            - one column per ticker (AAPL, MSFT, TSLA, ...)
    """
    close_df = extract_close_prices(prices_df)

    # Compute percentage change for each asset
    returns = close_df.pct_change()

    # Remove the first row (NaN)
    returns = returns.dropna(how="all")

    # Put timestamp back as a column
    returns = returns.reset_index().rename(columns={"index": "timestamp"})

    return returns


def compute_portfolio_equity(
    returns_df: pd.DataFrame,
    weights,
    initial_capital: float = 1.0,
    rebal_freq: str = "none",
) -> pd.DataFrame:
    """
    Simulate a multi-asset portfolio over time.

    Inputs:
        returns_df:
            - column "timestamp"
            - other columns = asset returns (one column per ticker)
        weights:
            - list or numpy array with one weight per asset
            - example: [0.4, 0.3, 0.3]
            - the order of weights must match the order of asset columns
        initial_capital:
            - starting portfolio value (e.g. 1.0 or 10000)
        rebal_freq:
            - "none"    : Buy & Hold (no rebalancing)
            - "weekly"  : rebalance at the first date of each week
            - "monthly" : rebalance at the first date of each month

    Output:
        out_df:
            - timestamp
            - portfolio_value   : portfolio value at each time step
            - portfolio_returns : portfolio return at each time step
    """
    if "timestamp" not in returns_df.columns:
        raise ValueError("returns_df must contain a 'timestamp' column.")

    # Sort by time
    df = returns_df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp")

    # Asset columns = all columns except "timestamp"
    asset_cols = [c for c in df.columns if c != "timestamp"]
    if len(asset_cols) == 0:
        raise ValueError("returns_df must contain at least one asset column.")

    # Convert weights to numpy array and normalize them
    weights = np.array(weights, dtype=float)
    if len(weights) != len(asset_cols):
        raise ValueError("Number of weights must be equal to number of assets.")
    if weights.sum() == 0:
        raise ValueError("Sum of weights must be > 0.")
    weights = weights / weights.sum()  # normalize so that sum(weights) = 1

    # Put timestamp as index and keep only the returns
    df = df.set_index("timestamp")
    df = df[asset_cols].astype(float)
    df = df.dropna(how="all")

    dates = df.index
    returns_matrix = df.values  # shape (T, N)

    # Compute rebalancing dates
    if rebal_freq == "none":
        rebal_dates = set()  # no rebalancing
    elif rebal_freq == "weekly":
        # First observed date of each week
        week_periods = dates.to_period("W")
        rebal_dates = set(dates.groupby(week_periods).head(1))
    elif rebal_freq == "monthly":
        # First observed date of each month
        month_periods = dates.to_period("M")
        rebal_dates = set(dates.groupby(month_periods).head(1))
    else:
        raise ValueError("rebal_freq must be 'none', 'weekly' or 'monthly'.")

    # Initialize portfolio
    portfolio_value = initial_capital
    current_weights = weights.copy()

    timestamps = []
    portfolio_values = []
    portfolio_returns = []

    for i, (ts, row_returns) in enumerate(zip(dates, returns_matrix)):
        if i == 0:
            # First point: no return yet
            timestamps.append(ts)
            portfolio_values.append(portfolio_value)
            portfolio_returns.append(0.0)
            continue

        # Portfolio return = dot product between weights and asset returns
        r_t = float(np.dot(current_weights, row_returns))

        # Update portfolio value
        portfolio_value = portfolio_value * (1.0 + r_t)

        timestamps.append(ts)
        portfolio_values.append(portfolio_value)
        portfolio_returns.append(r_t)

        # Update weights with "drift" (no transaction)
        asset_values = current_weights * (1.0 + row_returns)
        total_value = asset_values.sum()
        if total_value > 0:
            current_weights = asset_values / total_value

        # Rebalance if current date is a rebalancing date
        if ts in rebal_dates:
            # Reset weights to the original target weights
            current_weights = weights.copy()

    out_df = pd.DataFrame({
        "timestamp": timestamps,
        "portfolio_value": portfolio_values,
        "portfolio_returns": portfolio_returns,
    })

    return out_df


def compute_portfolio_stats(
    returns_df: pd.DataFrame,
    weights,
    periods_per_year: int = 252,
) -> dict:
    """
    Compute basic statistics for the portfolio:

        - annualized mean return of each asset
        - annualized volatility of each asset
        - annualized covariance matrix
        - correlation matrix
        - annualized mean return of the portfolio
        - annualized volatility of the portfolio
        - simple Sharpe ratio (risk-free rate = 0)

    Inputs:
        returns_df:
            - column "timestamp"
            - other columns = asset returns
        weights:
            - list or numpy array, one weight per asset
        periods_per_year:
            - 252 for daily data
            - 365 for daily calendar data
            - for hourly data, you can adapt this value

    Output:
        stats: a dictionary with pandas objects and simple floats.
    """
    if "timestamp" not in returns_df.columns:
        raise ValueError("returns_df must contain a 'timestamp' column.")

    df = returns_df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp")

    asset_cols = [c for c in df.columns if c != "timestamp"]
    if len(asset_cols) == 0:
        raise ValueError("returns_df must contain at least one asset column.")

    df = df.set_index("timestamp")
    asset_returns = df[asset_cols].astype(float).dropna(how="all")

    # Convert weights to numpy array and normalize
    weights = np.array(weights, dtype=float)
    if len(weights) != len(asset_cols):
        raise ValueError("Number of weights must be equal to number of assets.")
    if weights.sum() == 0:
        raise ValueError("Sum of weights must be > 0.")
    weights = weights / weights.sum()

    # Annualized mean and volatility for each asset
    mean_returns = asset_returns.mean() * periods_per_year
    vol_returns = asset_returns.std() * np.sqrt(periods_per_year)

    # Annualized covariance and correlation matrices
    cov_matrix = asset_returns.cov() * periods_per_year
    corr_matrix = asset_returns.corr()

    # Portfolio annualized mean and volatility
    mu_p = float(np.dot(weights, mean_returns.values))
    sigma_p = float(np.sqrt(np.dot(weights.T, np.dot(cov_matrix.values, weights))))

    # Simple Sharpe ratio with risk-free rate = 0
    if sigma_p > 0:
        sharpe_p = mu_p / sigma_p
    else:
        sharpe_p = np.nan

    stats = {
        "asset_mean_returns": mean_returns,
        "asset_volatility": vol_returns,
        "cov_matrix": cov_matrix,
        "corr_matrix": corr_matrix,
        "portfolio_mean_return": mu_p,
        "portfolio_volatility": sigma_p,
        "portfolio_sharpe": sharpe_p,
    }

    return stats
