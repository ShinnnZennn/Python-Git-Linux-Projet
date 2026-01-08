import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from scipy.stats import norm

# ============================================================
# RIDGE REGRESSION FORECAST
# ============================================================
def ridge_regression_forecast(
    df: pd.DataFrame,
    horizon: int = 30,
    confidence: float = 0.95,
    alpha: float = 10.0
) -> pd.DataFrame:

    y = df["close"].values
    t = np.arange(len(y))

    # Feature engineering
    X = np.column_stack([
        t,
        t**2,
        np.sin(t / 10),
        np.cos(t / 10),
    ])

    model = Ridge(alpha=alpha)
    model.fit(X, y)

    # Future features
    t_future = np.arange(len(y), len(y) + horizon)
    X_future = np.column_stack([
        t_future,
        t_future**2,
        np.sin(t_future / 10),
        np.cos(t_future / 10),
    ])

    predictions = model.predict(X_future)

    # Residuals and confidence interval
    fitted = model.predict(X)
    residuals = y - fitted
    sigma = residuals.std()

    z = norm.ppf(0.5 + confidence / 2)

    lower = predictions - z * sigma
    upper = predictions + z * sigma

    # Future timestamps
    last_ts = df["timestamp"].iloc[-1]
    freq = pd.infer_freq(df["timestamp"])

    future_dates = pd.date_range(
        start=last_ts,
        periods=horizon + 1,
        freq=freq,
        tz=last_ts.tz
    )[1:]

    return pd.DataFrame({
        "timestamp": future_dates,
        "prediction": predictions,
        "lower_ci": lower,
        "upper_ci": upper,
    })
