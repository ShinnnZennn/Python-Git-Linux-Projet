from single_asset.metrics import (
    compute_total_return,
    compute_annualized_return,
    compute_annualized_volatility,
    compute_sharpe_ratio,
    compute_max_drawdown
)


def compute_portfolio_metrics(
    portfolio_equity,
    periods_per_year: int
) -> dict:
    """
    Compute standard performance metrics for a portfolio equity curve.
    """

    metrics = {
        "total_return": compute_total_return(portfolio_equity),
        "annualized_return": compute_annualized_return(portfolio_equity, periods_per_year),
        "annualized_volatility": compute_annualized_volatility(portfolio_equity, periods_per_year),
        "sharpe_ratio": compute_sharpe_ratio(portfolio_equity, periods_per_year),
        "max_drawdown": compute_max_drawdown(portfolio_equity),
    }

    return metrics