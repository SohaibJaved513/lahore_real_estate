from __future__ import annotations

import time

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, root_mean_squared_error


def regression_metrics(y_true, y_pred) -> dict[str, float]:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    rmse = root_mean_squared_error(y_true, y_pred)
    return {
        "rmse_pkr": float(rmse),
        "rmse_percent_mean": float(rmse / np.mean(y_true) * 100),
        "mape_percent": float(mean_absolute_percentage_error(y_true, y_pred) * 100),
        "mae_pkr": float(mean_absolute_error(y_true, y_pred)),
    }


def median_latency_ms(model, sample, runs: int = 100) -> float:
    durations = []
    for _ in range(runs):
        start = time.perf_counter()
        model.predict(sample)
        durations.append((time.perf_counter() - start) * 1000)
    return float(np.median(durations))


def deployment_latency(model, single_sample, batch_sample) -> dict[str, float]:
    return {
        "single_prediction_ms": median_latency_ms(model, single_sample, runs=1000),
        "batch_100_ms": median_latency_ms(model, batch_sample, runs=100),
    }
