from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import TransformedTargetRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor

from realestate.config import RANDOM_STATE
from realestate.features import build_preprocessor


def price_bins(y: pd.Series, bins: int = 10) -> pd.Series:
    count = min(bins, max(2, len(y) // 50))
    return pd.qcut(y.rank(method="first"), q=count, labels=False, duplicates="drop")


def validation_splits(X: pd.DataFrame, y: pd.Series):
    splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    return list(splitter.split(X, price_bins(y)))


def xgb_pipeline(feature_columns: list[str] | None = None, **params) -> Pipeline:
    defaults = {
        "n_estimators": 260,
        "max_depth": 5,
        "learning_rate": 0.05,
        "subsample": 0.9,
        "colsample_bytree": 0.9,
        "min_child_weight": 2,
        "reg_alpha": 0.0,
        "reg_lambda": 1.2,
        "objective": "reg:squarederror",
        "tree_method": "hist",
        "random_state": RANDOM_STATE,
        "n_jobs": 2,
    }
    defaults.update(params)
    target_model = TransformedTargetRegressor(
        regressor=XGBRegressor(**defaults),
        func=np.log1p,
        inverse_func=np.expm1,
    )
    return Pipeline([
        ("preprocessor", build_preprocessor(feature_columns)),
        ("model", target_model),
    ])


def ridge_pipeline(feature_columns: list[str] | None = None, alpha: float = 10.0) -> Pipeline:
    target_model = TransformedTargetRegressor(
        regressor=Ridge(alpha=alpha, solver="lsqr"),
        func=np.log1p,
        inverse_func=np.expm1,
    )
    return Pipeline([
        ("preprocessor", build_preprocessor(feature_columns)),
        ("model", target_model),
    ])


def tune_xgb(X: pd.DataFrame, y: pd.Series, output_csv: Path, quick: bool = False):
    parameters = {
        "model__regressor__n_estimators": [160, 240, 320, 420],
        "model__regressor__max_depth": [3, 4, 5, 6],
        "model__regressor__learning_rate": [0.03, 0.05, 0.08, 0.12],
        "model__regressor__subsample": [0.75, 0.9, 1.0],
        "model__regressor__colsample_bytree": [0.75, 0.9, 1.0],
        "model__regressor__min_child_weight": [1, 2, 4, 6],
        "model__regressor__reg_alpha": [0.0, 0.05, 0.2],
        "model__regressor__reg_lambda": [0.8, 1.2, 2.0, 3.0],
    }
    search = RandomizedSearchCV(
        xgb_pipeline(),
        parameters,
        n_iter=2 if quick else 10,
        scoring="neg_root_mean_squared_error",
        cv=validation_splits(X, y),
        random_state=RANDOM_STATE,
        n_jobs=1,
        refit=True,
        return_train_score=True,
    )
    search.fit(X, y)
    log = pd.DataFrame(search.cv_results_)
    log["validation_rmse_pkr"] = -log["mean_test_score"]
    log["validation_rmse_percent_mean"] = log["validation_rmse_pkr"] / y.mean() * 100
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    log.to_csv(output_csv, index=False)
    return search.best_estimator_, log, search.best_params_


def tune_ridge(X: pd.DataFrame, y: pd.Series, output_csv: Path, quick: bool = False):
    alpha_values = [1.0, 10.0] if quick else [0.1, 1.0, 5.0, 10.0, 25.0, 50.0, 100.0]
    search = GridSearchCV(
        ridge_pipeline(),
        {"model__regressor__alpha": alpha_values},
        scoring="neg_root_mean_squared_error",
        cv=validation_splits(X, y),
        n_jobs=1,
        refit=True,
        return_train_score=True,
    )
    search.fit(X, y)
    log = pd.DataFrame(search.cv_results_)
    log["validation_rmse_pkr"] = -log["mean_test_score"]
    log["validation_rmse_percent_mean"] = log["validation_rmse_pkr"] / y.mean() * 100
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    log.to_csv(output_csv, index=False)
    return search.best_estimator_, log, search.best_params_


def save_json(data: dict | list, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(data, file, indent=2, default=float)


def save_model(model: Pipeline, path: Path) -> float:
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path, compress=3)
    return path.stat().st_size / (1024 * 1024)
