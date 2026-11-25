from __future__ import annotations

import numpy as np


def calibrate_interval(y_true, y_pred, coverage: float = 0.90) -> float:
    true_log = np.log1p(np.asarray(y_true))
    pred_log = np.log1p(np.maximum(np.asarray(y_pred), 0))
    residuals = np.abs(true_log - pred_log)
    level = min(1.0, np.ceil((len(residuals) + 1) * coverage) / len(residuals))
    return float(np.quantile(residuals, level, method="higher"))


def prediction_interval(predictions, radius_log: float) -> tuple[np.ndarray, np.ndarray]:
    center = np.log1p(np.maximum(np.asarray(predictions), 0))
    lower = np.expm1(center - radius_log)
    upper = np.expm1(center + radius_log)
    return np.maximum(0, lower), upper


def empirical_coverage(y_true, lower, upper) -> float:
    y_true = np.asarray(y_true)
    return float(np.mean((y_true >= lower) & (y_true <= upper)))
