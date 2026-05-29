"""
Forecasting utility for sentiment trend prediction using linear regression.
Uses only non-zero data points so sparse location data does not skew the fit.
"""

import numpy as np
from datetime import timedelta, datetime


def linear_regression_forecast(sentiment_data, days_ahead=7):
    """
    Perform linear regression on sentiment data and forecast future values.
    Only non-zero values are used for the regression so that days with no
    tweet activity do not artificially drag the trend line toward zero.

    Args:
        sentiment_data: List of sentiment percentages (0-100) per day.
                        Zeros mean "no data that day", not 0 % negative.
        days_ahead: Number of days to forecast (default 7)

    Returns:
        dict with keys:
            forecast, confidence_upper, confidence_lower,
            slope, trend_direction, r_squared,
            has_sufficient_data, warning
        Returns a dict with empty forecast lists when data is insufficient.
    """
    if not sentiment_data:
        return _insufficient(0, 7)

    data_array = np.array(sentiment_data, dtype=float)

    # Use only days that actually had tweets (non-zero sentiment values)
    valid_indices = np.where(data_array > 0)[0]
    n_valid = len(valid_indices)

    if n_valid < 7:
        return _insufficient(n_valid, 7)

    # --- Regression on valid points only ---
    x = valid_indices.astype(float)   # x = position in the full timeline
    y = data_array[valid_indices]     # y = actual negative % on those days

    coefficients = np.polyfit(x, y, 1)
    m, b = coefficients[0], coefficients[1]

    # R² on the training data
    y_pred_train = m * x + b
    ss_res = np.sum((y - y_pred_train) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r_squared = float(1 - ss_res / ss_tot) if ss_tot != 0 else 0.0

    # Residual standard deviation for confidence intervals
    residuals   = y - y_pred_train
    std_residual = float(np.std(residuals)) if len(residuals) > 1 else 1.0

    # --- Forecast starting right after the last observed day ---
    last_x      = int(valid_indices[-1])
    forecast_x  = np.arange(last_x + 1, last_x + 1 + days_ahead, dtype=float)
    forecast_y  = np.clip(m * forecast_x + b, 0, 100)

    # 95 % confidence intervals (widening with forecast distance)
    x_mean   = float(np.mean(x))
    x_ss     = float(np.sum((x - x_mean) ** 2)) + 1e-10
    std_errs = std_residual * np.sqrt(
        1 + 1 / n_valid + (forecast_x - x_mean) ** 2 / x_ss
    )
    z = 1.96
    ci_upper = np.clip(forecast_y + z * std_errs, 0, 100)
    ci_lower = np.clip(forecast_y - z * std_errs, 0, 100)

    trend_direction   = get_trend_direction(m)
    has_sufficient    = n_valid >= 14
    warning           = (
        None if has_sufficient else
        f'Limited data ({n_valid} valid data points) — forecast may be less accurate'
    )

    return {
        'forecast':          forecast_y.tolist(),
        'confidence_upper':  ci_upper.tolist(),
        'confidence_lower':  ci_lower.tolist(),
        'slope':             round(float(m), 2),
        'trend_direction':   trend_direction,
        'r_squared':         round(r_squared, 3),
        'has_sufficient_data': has_sufficient,
        'warning':           warning,
    }


def _insufficient(n_valid, needed):
    """Return a no-forecast result with a clear message."""
    still_needed = max(0, needed - n_valid)
    msg = (
        f'Only {n_valid} data point{"s" if n_valid != 1 else ""} available — '
        f'need at least {needed} ({still_needed} more needed)'
    )
    return {
        'forecast': [], 'confidence_upper': [], 'confidence_lower': [],
        'slope': 0, 'trend_direction': 'STABLE', 'r_squared': 0,
        'has_sufficient_data': False, 'warning': msg,
    }


def get_trend_direction(slope):
    """
    RISING  if slope >  2 % / day
    FALLING if slope < -2 % / day
    STABLE  otherwise
    """
    if slope > 2:
        return 'RISING'
    elif slope < -2:
        return 'FALLING'
    return 'STABLE'


def get_next_7_dates(start_date=None):
    """Return the next 7 calendar dates as YYYY-MM-DD strings."""
    if start_date is None:
        start_date = datetime.now().date()
    return [
        (start_date + timedelta(days=i + 1)).strftime('%Y-%m-%d')
        for i in range(7)
    ]
