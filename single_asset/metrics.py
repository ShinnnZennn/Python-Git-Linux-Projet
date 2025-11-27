import pandas as pd
import numpy as np


def compute_total_return(series: pd.Series) -> float:
    """Return Total: (last / first) - 1"""
    return float(series.iloc[-1] / series.iloc[0] - 1)


def compute_annualized_return(series: pd.Series, periods_per_year: int) -> float:
    """Return annualisé"""
    total_return = compute_total_return(series)
    n_periods = len(series)
    return float((1 + total_return) ** (periods_per_year / n_periods) - 1)


def compute_annualized_volatility(returns: pd.Series, periods_per_year: int) -> float:
    """Volatilité annualisée"""
    return float(returns.std() * np.sqrt(periods_per_year))


def compute_sharpe_ratio(returns: pd.Series, periods_per_year: int, risk_free_rate=0.0) -> float:
    """Sharpe ratio (RF = 0 par défaut)"""
    ann_ret = compute_annualized_return((1 + returns).cumprod(), periods_per_year)
    ann_vol = compute_annualized_volatility(returns, periods_per_year)

    if ann_vol == 0:
        return np.nan

    return float((ann_ret - risk_free_rate) / ann_vol)


def compute_max_drawdown(series: pd.Series) -> float:
    """Max drawdown"""
    cummax = series.cummax()
    drawdown = (series - cummax) / cummax
    return float(drawdown.min())
