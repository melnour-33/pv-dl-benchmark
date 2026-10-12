"""Torch-free ARIMA worker. Kept in its own module so that each parallel
worker process imports only numpy/pmdarima (not torch), which keeps the
per-worker memory small on many-core HPC nodes."""
import warnings

import numpy as np
from pmdarima import auto_arima
from pmdarima.arima import ARIMA


# ---------------------------------------------------------------------------
# Module-level worker (must be picklable for joblib/loky on Windows & Linux).
# Fits one ARIMA on one window and returns a forecast of length `n_periods`.
# The fitting settings are identical to the previous serial implementation.
# ---------------------------------------------------------------------------
def _fit_forecast_one(series, exog, order, n_periods):
    warnings.filterwarnings('ignore')
    series = np.asarray(series, dtype=np.float64).flatten()
    last_val = series[-1] if series.size > 0 else 0.0
    if series.size < 3:
        return np.full(n_periods, last_val)

    try:
        if order is not None:
            model = ARIMA(order=order).fit(series, X=exog)
        else:
            model = auto_arima(
                series,
                X=exog,
                start_p=0, start_q=0,
                max_p=5, max_q=5,
                seasonal=False,
                stepwise=True,
                suppress_warnings=True,
                error_action='raise',
                max_order=None,
                information_criterion='aic'
            )
    except Exception:
        return np.full(n_periods, last_val)

    try:
        if exog is not None:
            # future exogenous values unknown -> hold last observed row
            exog_future = np.tile(exog[-1:, :], (n_periods, 1))
            return np.asarray(model.predict(n_periods=n_periods, X=exog_future))
        return np.asarray(model.predict(n_periods=n_periods))
    except Exception:
        return np.full(n_periods, last_val)
