from __future__ import annotations

import argparse
import time

import pandas as pd
from sklearn.model_selection import train_test_split

from realestate.config import (
    ARTIFACT_DIR,
    CALIBRATION_SIZE,
    CLEANING_AUDIT_PATH,
    FEATURES_PATH,
    INTERVAL_PATH,
    METRICS_PATH,
    MODEL_PATH,
    PLOTS_DIR,
    PROCESSED_DATA_PATH,
    RANDOM_STATE,
    RESULTS_DIR,
    TABLES_DIR,
    TEST_SIZE,
    TUNING_DIR,
    UI_OPTIONS_PATH,
)
from realestate.data import load_property_data
from realestate.evaluation import (
    amenity_correlation_plot,
    interval_plot,
    permutation_table,
    predicted_actual_plot,
    residual_plots,
    search_sensitivity_note,
    shap_table,
    stress_tests,
    tuning_sensitivity_plot,
    write_report,
)
from realestate.features import MODEL_FEATURES, PROTECTED_FEATURES, make_features
from realestate.metrics import deployment_latency, median_latency_ms, regression_metrics
from realestate.modeling import (
    price_bins,
    ridge_pipeline,
    save_json,
    save_model,
    tune_ridge,
    tune_xgb,
    xgb_pipeline,
)
from realestate.uncertainty import calibrate_interval, empirical_coverage, prediction_interval


def ensure_directories() -> None:
    for path in [ARTIFACT_DIR, PLOTS_DIR, TABLES_DIR, PROCESSED_DATA_PATH.parent, TUNING_DIR]:
        path.mkdir(parents=True, exist_ok=True)


def split_data(data: pd.DataFrame):
    train, test = train_test_split(
        data,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=price_bins(data["price"]),
    )
    model_train, calibration = train_test_split(
        train,
        test_size=CALIBRATION_SIZE,
        random_state=RANDOM_STATE,
        stratify=price_bins(train["price"]),
    )
    return tuple(part.reset_index(drop=True) for part in (model_train, calibration, test))


def model_row(name, model, X_train, y_train, X_test, y_test):
    start = time.perf_counter()
    model.fit(X_train, y_train)
    train_time = time.perf_counter() - start
    predictions = model.predict(X_test)
    row = regression_metrics(y_test, predictions)
    row.update({
        "model": name,
        "train_time_seconds": round(train_time, 3),
        "latency_ms_per_prediction": round(median_latency_ms(model, X_test.iloc[[0]], runs=40), 3),
        "final": "No",
    })
    return row, predictions, model


def build_family_model(family: str, features: list[str], xgb_params: dict, ridge_params: dict):
    if family == "ridge":
        alpha = ridge_params.get("model__regressor__alpha", 10.0)
        return ridge_pipeline(features, alpha=float(alpha))
    params = {key.replace("model__regressor__", ""): value for key, value in xgb_params.items()}
    return xgb_pipeline(features, **params)


def save_data_profile(raw: pd.DataFrame, audit: dict) -> None:
    profile = [
        {"measure": "Raw rows", "value": audit["initial_rows"]},
        {"measure": "Clean modeling rows", "value": audit["final_rows"]},
        {"measure": "Unique full locations", "value": raw["location"].nunique()},
        {"measure": "Unique broad societies", "value": raw["city_area"].nunique()},
        {"measure": "Median price PKR", "value": raw["price"].median()},
        {"measure": "Median area sqft", "value": raw["area_sqft"].median()},
        {"measure": "Missing built year", "value": int(raw["built_year"].isna().sum())},
        {"measure": "Missing bedrooms", "value": int(raw["bedrooms"].isna().sum())},
        {"measure": "Missing bathrooms", "value": int(raw["bathrooms"].isna().sum())},
    ]
    pd.DataFrame(profile).to_csv(TABLES_DIR / "data_quality_summary.csv", index=False)


def run_workflow(data_path: str | None = None, quick: bool | None = None) -> dict:
    ensure_directories()
    if quick is None:
        parser = argparse.ArgumentParser()
        parser.add_argument("--data", default=None)
        parser.add_argument("--quick", action="store_true")
        args = parser.parse_args()
        data_path = args.data
        quick = args.quick

    raw, data_mode, audit = load_property_data(data_path)
    data = make_features(raw)
    if quick and len(data) > 1800:
        data = data.sample(1800, random_state=RANDOM_STATE).reset_index(drop=True)
        audit = {**audit, "quick_mode_rows": len(data)}
    data.to_csv(PROCESSED_DATA_PATH, index=False)
    save_json(audit, CLEANING_AUDIT_PATH)
    save_data_profile(raw, audit)

    ui_options = {
        "locations": raw["location"].value_counts().head(120).index.tolist(),
        "city_areas": raw["city_area"].value_counts().head(80).index.tolist(),
        "dataset_scope": "Lahore houses for sale",
    }
    save_json(ui_options, UI_OPTIONS_PATH)

    model_train, calibration, test = split_data(data)
    X_train = model_train[MODEL_FEATURES]
    y_train = model_train["price"]
    X_cal = calibration[MODEL_FEATURES]
    y_cal = calibration["price"]
    X_test = test[MODEL_FEATURES]
    y_test = test["price"]

    untuned_row, _, _ = model_row(
        "XGBoost untuned",
        xgb_pipeline(n_estimators=180, max_depth=4, learning_rate=0.06),
        X_train,
        y_train,
        X_test,
        y_test,
    )

    tuned_search_model, xgb_log, xgb_params = tune_xgb(
        X_train,
        y_train,
        TUNING_DIR / "trial_log.csv",
        quick=bool(quick),
    )
    tuned_row, tuned_predictions, tuned_model = model_row(
        "XGBoost tuned",
        tuned_search_model,
        X_train,
        y_train,
        X_test,
        y_test,
    )

    ridge_search_model, _, ridge_params = tune_ridge(
        X_train,
        y_train,
        TUNING_DIR / "ridge_trial_log.csv",
        quick=bool(quick),
    )
    ridge_row, ridge_predictions, ridge_model = model_row(
        "Ridge tuned",
        ridge_search_model,
        X_train,
        y_train,
        X_test,
        y_test,
    )

    blend_predictions = 0.85 * tuned_predictions + 0.15 * ridge_predictions
    blend_row = regression_metrics(y_test, blend_predictions)
    blend_row.update({
        "model": "Weighted blend",
        "train_time_seconds": round(tuned_row["train_time_seconds"] + ridge_row["train_time_seconds"], 3),
        "latency_ms_per_prediction": round(tuned_row["latency_ms_per_prediction"] + ridge_row["latency_ms_per_prediction"], 3),
        "final": "Comparison only",
    })

    if ridge_row["rmse_pkr"] < tuned_row["rmse_pkr"]:
        winner_family = "ridge"
        winner_name = "Ridge tuned"
        winner_model = ridge_model
        winner_row = ridge_row
        winner_predictions = ridge_predictions
    else:
        winner_family = "xgb"
        winner_name = "XGBoost tuned"
        winner_model = tuned_model
        winner_row = tuned_row
        winner_predictions = tuned_predictions

    comparison = [untuned_row, tuned_row, ridge_row, blend_row]
    for row in comparison:
        if row["model"] == winner_name:
            row["final"] = "Candidate"
    pd.DataFrame(comparison).to_csv(TABLES_DIR / "model_comparison.csv", index=False)
    save_json({"xgboost": xgb_params, "ridge": ridge_params}, TUNING_DIR / "best_config.json")
    tuning_sensitivity_plot(xgb_log, PLOTS_DIR / "hyperparameter_sensitivity.png")

    permutation = permutation_table(
        winner_model,
        X_test,
        y_test,
        TABLES_DIR / "permutation_importance.csv",
        PLOTS_DIR / "permutation_importance.png",
    )
    shap_table(
        winner_model,
        X_test,
        TABLES_DIR / "shap_importance.csv",
        PLOTS_DIR / "shap_importance.png",
    )

    removable = [
        feature
        for feature in permutation.sort_values("importance_rmse_pkr")["feature"].tolist()
        if feature not in PROTECTED_FEATURES
    ]
    drop_count = max(1, len(MODEL_FEATURES) // 5)
    removed_features = removable[:drop_count]
    trimmed_features = [feature for feature in MODEL_FEATURES if feature not in removed_features]
    trimmed_candidate = build_family_model(winner_family, trimmed_features, xgb_params, ridge_params)
    trimmed_row, trimmed_predictions, trimmed_candidate = model_row(
        f"Trimmed {winner_name}",
        trimmed_candidate,
        X_train[trimmed_features],
        y_train,
        X_test[trimmed_features],
        y_test,
    )
    feature_delta = (trimmed_row["rmse_pkr"] - winner_row["rmse_pkr"]) / winner_row["rmse_pkr"] * 100
    feature_selection = [
        {
            "version": "Full feature set",
            "feature_count": len(MODEL_FEATURES),
            "removed_features": "None",
            "rmse_pkr": winner_row["rmse_pkr"],
            "rmse_change_percent": 0.0,
        },
        {
            "version": "Trimmed feature set",
            "feature_count": len(trimmed_features),
            "removed_features": ", ".join(removed_features),
            "rmse_pkr": trimmed_row["rmse_pkr"],
            "rmse_change_percent": feature_delta,
        },
    ]
    pd.DataFrame(feature_selection).to_csv(TABLES_DIR / "feature_selection_comparison.csv", index=False)

    full_temp = ARTIFACT_DIR / "full_model_temp.joblib"
    trimmed_temp = ARTIFACT_DIR / "trimmed_model_temp.joblib"
    full_size = save_model(winner_model, full_temp)
    trimmed_size = save_model(trimmed_candidate, trimmed_temp)

    test_batch_full = X_test.iloc[: min(100, len(X_test))]
    test_batch_trimmed = X_test[trimmed_features].iloc[: min(100, len(X_test))]
    full_latency = deployment_latency(winner_model, X_test.iloc[[0]], test_batch_full)
    trimmed_latency = deployment_latency(
        trimmed_candidate,
        X_test[trimmed_features].iloc[[0]],
        test_batch_trimmed,
    )

    size_gain_percent = (full_size - trimmed_size) / full_size * 100
    latency_gain_percent = (
        full_latency["single_prediction_ms"] - trimmed_latency["single_prediction_ms"]
    ) / full_latency["single_prediction_ms"] * 100
    use_trimmed = feature_delta <= 3.0 and (
        feature_delta < 0 or size_gain_percent >= 5.0 or latency_gain_percent >= 5.0
    )
    final_model = trimmed_candidate if use_trimmed else winner_model
    final_features = trimmed_features if use_trimmed else MODEL_FEATURES
    final_predictions = trimmed_predictions if use_trimmed else winner_predictions
    selected_name = f"Trimmed {winner_name}" if use_trimmed else winner_name
    final_latency = trimmed_latency if use_trimmed else full_latency
    final_size = save_model(final_model, MODEL_PATH)
    full_temp.unlink(missing_ok=True)
    trimmed_temp.unlink(missing_ok=True)

    final_cal_predictions = final_model.predict(X_cal[final_features])
    radius_log = calibrate_interval(y_cal, final_cal_predictions, coverage=0.90)
    lower, upper = prediction_interval(final_predictions, radius_log)
    coverage = empirical_coverage(y_test, lower, upper)
    final_metrics = regression_metrics(y_test, final_predictions)
    median_width = float(pd.Series(upper - lower).median())

    predicted_actual_plot(y_test, final_predictions, PLOTS_DIR / "predicted_vs_actual.png")
    residual_plots(test, y_test, final_predictions, PLOTS_DIR)
    interval_plot(y_test, final_predictions, lower, upper, PLOTS_DIR / "prediction_intervals.png")
    amenity_correlation_plot(test, PLOTS_DIR / "amenity_correlation.png")

    test_context = make_features(test)
    train_location_counts = model_train["location"].value_counts()
    stress = stress_tests(
        final_model,
        X_test[final_features],
        y_test,
        test_context,
        train_location_counts,
    )
    stress.to_csv(TABLES_DIR / "stress_scenarios.csv", index=False)

    optimization = [
        {
            "version": f"Full {winner_name}",
            "artifact_size_mb": full_size,
            **full_latency,
            "rmse_pkr": winner_row["rmse_pkr"],
        },
        {
            "version": f"Trimmed {winner_name}",
            "artifact_size_mb": trimmed_size,
            **trimmed_latency,
            "rmse_pkr": trimmed_row["rmse_pkr"],
        },
        {
            "version": "Selected production model",
            "artifact_size_mb": final_size,
            **final_latency,
            "rmse_pkr": final_metrics["rmse_pkr"],
        },
    ]
    pd.DataFrame(optimization).to_csv(TABLES_DIR / "optimization_tradeoff.csv", index=False)
    optimization_markdown = pd.DataFrame(optimization).to_markdown(index=False)
    (RESULTS_DIR.parent / "deployment" / "OPTIMIZATION_REPORT.md").write_text(
        "# Deployment Optimization Report\n\n"
        + optimization_markdown
        + "\n\nThe feature-trimmed model is used only when held-out RMSE remains within the 3% tolerance. "
        + "Single-listing latency is the median across 1,000 runs; batch latency is the median across 100 runs.\n",
        encoding="utf-8",
    )

    save_json({
        "method": "split_conformal_log_residual",
        "coverage_target": 0.90,
        "radius_log": radius_log,
        "empirical_coverage": coverage,
        "median_interval_width_pkr": median_width,
    }, INTERVAL_PATH)
    save_json({
        "data_mode": data_mode,
        "dataset_file": "lahore_house_listings_zameen.csv",
        "selected_model": selected_name,
        "test_records": len(test),
        "training_records": len(model_train),
        "calibration_records": len(calibration),
        "clean_records": len(data),
        "metrics": final_metrics,
        "interval_coverage_percent": coverage * 100,
        "median_interval_width_pkr": median_width,
        "artifact_size_mb": final_size,
        **final_latency,
    }, METRICS_PATH)
    save_json({
        "model_features": final_features,
        "removed_features": removed_features if use_trimmed else [],
        "scope": "Lahore houses for sale",
    }, FEATURES_PATH)

    context = {
        "data_mode": data_mode,
        "cleaning_audit": audit,
        "comparison": comparison,
        "selected_model": selected_name,
        "coverage_percent": coverage * 100,
        "median_interval_width_pkr": median_width,
        "feature_selection": feature_selection,
        "stress_tests": stress.to_dict(orient="records"),
        "optimization": optimization,
        "tuning_note": search_sensitivity_note(xgb_log, xgb_params),
    }
    write_report(context, RESULTS_DIR / "REPORT.md")
    return context
