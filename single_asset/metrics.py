import pandas as pd
import numpy as np

def max_drawdown(cum: pd.Series) -> float:
    rolling_max = cum.cummax()
    drawdown = cum / rolling_max - 1
    return drawdown.min()

def annualized_vol(returns: pd.Series, periods_per_year: int = 252) -> float:
    return returns.std() * np.sqrt(periods_per_year)

def sharpe_ratio(returns: pd.Series, rf: float = 0.0, periods_per_year: int = 252) -> float:
    excess = returns - rf / periods_per_year
    vol = annualized_vol(excess, periods_per_year)
    return 0 if vol == 0 else (excess.mean() * periods_per_year / vol)
