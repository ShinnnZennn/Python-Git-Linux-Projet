import pandas as pd
import numpy as np


def compute_total_return(series: pd.Series) -> float:
    """Total return : (final / initial) - 1"""
    return float(series.iloc[-1] / series.iloc[0] - 1)


def compute_log_returns(series: pd.Series) -> pd.Series:
    """Log returns"""
    return np.log(series / series.shift(1)).fillna(0.0)


def compute_annualized_return(series: pd.Series, periods_per_year: int) -> float:
    """
    Annualized return using log returns:
    exp(mean(log_ret) * periods_per_year) - 1
    """
    log_ret = compute_log_returns(series)
    mean_log_ret = log_ret.mean()
    return float(np.exp(mean_log_ret * periods_per_year) - 1)


def compute_annualized_volatility(series: pd.Series, periods_per_year: int) -> float:
    """
    Annualized volatility using log returns:
    std(log_ret) * sqrt(periods_per_year)
    """
    log_ret = compute_log_returns(series)
    return float(log_ret.std() * np.sqrt(periods_per_year))


def compute_sharpe_ratio(series: pd.Series, periods_per_year: int, risk_free_rate: float = 0.0) -> float:
    """
    Sharpe ratio:
    (annual_return - rf) / annual_vol
    RF = 0 by default
    """
    ann_ret = compute_annualized_return(series, periods_per_year)
    ann_vol = compute_annualized_volatility(series, periods_per_year)

    if ann_vol == 0:
        return np.nan

    return float((ann_ret - risk_free_rate) / ann_vol)


def compute_max_drawdown(series: pd.Series) -> float:
    """Max drawdown: min((equity - cummax) / cummax)"""
    cummax = series.cummax()
    drawdown = (series - cummax) / cummax
    return float(drawdown.min())
