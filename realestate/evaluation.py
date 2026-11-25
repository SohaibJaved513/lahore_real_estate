from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.inspection import permutation_importance

from realestate.features import AMENITY_FEATURES
from realestate.metrics import regression_metrics

CRORE = 10_000_000


def save_plot(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=180, bbox_inches="tight")
    plt.close()


def tuning_sensitivity_plot(search_log: pd.DataFrame, path: Path) -> None:
    depth = "param_model__regressor__max_depth"
    trees = "param_model__regressor__n_estimators"
    plt.figure(figsize=(8, 5))
    if depth in search_log and trees in search_log:
        points = plt.scatter(
            search_log[trees].astype(float),
            search_log["validation_rmse_pkr"] / CRORE,
            c=search_log[depth].astype(float),
            s=85,
            alpha=0.85,
        )
        plt.colorbar(points, label="Maximum depth")
        plt.xlabel("Number of estimators")
    else:
        plt.plot(search_log.index + 1, search_log["validation_rmse_pkr"] / CRORE, marker="o")
        plt.xlabel("Search trial")
    plt.ylabel("Validation RMSE (crore PKR)")
    plt.title("XGBoost Hyperparameter Search")
    save_plot(path)


def predicted_actual_plot(y_true, predictions, path: Path) -> None:
    actual = np.asarray(y_true) / CRORE
    predicted = np.asarray(predictions) / CRORE
    plt.figure(figsize=(7, 6))
    plt.scatter(actual, predicted, alpha=0.35, s=18)
    limit = max(float(actual.max()), float(predicted.max()))
    plt.plot([0, limit], [0, limit], linestyle="--", linewidth=1.2)
    plt.xlabel("Actual price (crore PKR)")
    plt.ylabel("Predicted price (crore PKR)")
    plt.title("Predicted vs Actual House Prices")
    save_plot(path)


def residual_plots(frame: pd.DataFrame, y_true, predictions, plots_dir: Path) -> None:
    residuals = (np.asarray(y_true) - np.asarray(predictions)) / CRORE

    plt.figure(figsize=(7, 5))
    plt.scatter(np.asarray(predictions) / CRORE, residuals, alpha=0.35, s=18)
    plt.axhline(0, linestyle="--", linewidth=1)
    plt.xlabel("Predicted price (crore PKR)")
    plt.ylabel("Residual (crore PKR)")
    plt.title("Residuals vs Predicted Price")
    save_plot(plots_dir / "residual_vs_prediction.png")

    plt.figure(figsize=(7, 5))
    plt.scatter(frame["area_sqft"], residuals, alpha=0.35, s=18)
    plt.axhline(0, linestyle="--", linewidth=1)
    plt.xlabel("Area (square feet)")
    plt.ylabel("Residual (crore PKR)")
    plt.title("Residuals vs Property Area")
    save_plot(plots_dir / "residual_vs_area.png")

    location_table = pd.DataFrame({"location": frame["city_area"], "residual": residuals})
    top_locations = location_table["location"].value_counts().head(10).index.tolist()
    groups = [location_table.loc[location_table["location"] == name, "residual"].values for name in top_locations]
    plt.figure(figsize=(11, 5))
    plt.boxplot(groups, tick_labels=top_locations, showfliers=False)
    plt.xticks(rotation=35, ha="right")
    plt.ylabel("Residual (crore PKR)")
    plt.title("Residual Distribution by Common Society")
    save_plot(plots_dir / "residual_by_location.png")


def interval_plot(y_true, predictions, lower, upper, path: Path) -> None:
    order = np.argsort(predictions)
    if len(order) > 100:
        order = order[np.linspace(0, len(order) - 1, 100, dtype=int)]
    x = np.arange(len(order))
    plt.figure(figsize=(11, 5))
    plt.fill_between(x, np.asarray(lower)[order] / CRORE, np.asarray(upper)[order] / CRORE, alpha=0.25, label="90% interval")
    plt.plot(x, np.asarray(predictions)[order] / CRORE, linewidth=1.2, label="Prediction")
    plt.scatter(x, np.asarray(y_true)[order] / CRORE, s=13, label="Actual")
    plt.xlabel("Listings sorted by predicted price")
    plt.ylabel("Price (crore PKR)")
    plt.title("Prediction Intervals on Held-out Listings")
    plt.legend()
    save_plot(path)


def amenity_correlation_plot(frame: pd.DataFrame, path: Path) -> None:
    columns = [column for column in AMENITY_FEATURES if column in frame]
    corr = frame[columns].corr().fillna(0)
    plt.figure(figsize=(9, 7))
    image = plt.imshow(corr, vmin=-1, vmax=1, cmap="coolwarm")
    plt.colorbar(image, fraction=0.046, pad=0.04)
    plt.xticks(range(len(columns)), columns, rotation=45, ha="right")
    plt.yticks(range(len(columns)), columns)
    plt.title("Amenity Flag Correlation Heatmap")
    save_plot(path)


def permutation_table(model, X, y, output_csv: Path, plot_path: Path) -> pd.DataFrame:
    result = permutation_importance(
        model,
        X,
        y,
        n_repeats=3,
        random_state=42,
        scoring="neg_root_mean_squared_error",
        n_jobs=1,
    )
    table = pd.DataFrame({
        "feature": X.columns,
        "importance_rmse_pkr": result.importances_mean,
        "std_pkr": result.importances_std,
    }).sort_values("importance_rmse_pkr", ascending=False)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(output_csv, index=False)
    shown = table.head(18).sort_values("importance_rmse_pkr")
    plt.figure(figsize=(8, 7))
    plt.barh(shown["feature"], shown["importance_rmse_pkr"] / CRORE)
    plt.xlabel("Increase in RMSE (crore PKR)")
    plt.title("Permutation Importance")
    save_plot(plot_path)
    return table


def shap_table(model, X: pd.DataFrame, output_csv: Path, plot_path: Path) -> pd.DataFrame:
    sample = X.iloc[: min(250, len(X))]
    preprocessor = model.named_steps["preprocessor"]
    transformed = preprocessor.transform(sample)
    feature_names = preprocessor.get_feature_names_out()
    estimator = model.named_steps["model"].regressor_

    if estimator.__class__.__module__.startswith("xgboost"):
        values = shap.TreeExplainer(estimator).shap_values(transformed)
    else:
        values = shap.LinearExplainer(estimator, transformed).shap_values(transformed)
    importance = np.abs(np.asarray(values)).mean(axis=0)
    table = pd.DataFrame({
        "transformed_feature": feature_names,
        "mean_abs_shap_log_price": importance,
    }).sort_values("mean_abs_shap_log_price", ascending=False)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(output_csv, index=False)
    shown = table.head(20).sort_values("mean_abs_shap_log_price")
    plt.figure(figsize=(8, 8))
    plt.barh(shown["transformed_feature"], shown["mean_abs_shap_log_price"])
    plt.xlabel("Mean absolute SHAP value (log-price space)")
    plt.title("SHAP Feature Importance")
    save_plot(plot_path)
    return table


def stress_tests(model, X: pd.DataFrame, y: pd.Series, context: pd.DataFrame, train_location_counts: pd.Series) -> pd.DataFrame:
    predictions = np.asarray(model.predict(X))
    y_values = y.reset_index(drop=True)
    context = context.reset_index(drop=True)
    tests = []

    scenarios = {
        "Top 1% price listings": y_values >= y_values.quantile(0.99),
        "Unusual area-to-bedroom ratio": (
            (context["area_sqft"] / context["bedrooms"].replace(0, np.nan)).lt(
                (context["area_sqft"] / context["bedrooms"].replace(0, np.nan)).quantile(0.05)
            )
            | (context["area_sqft"] / context["bedrooms"].replace(0, np.nan)).gt(
                (context["area_sqft"] / context["bedrooms"].replace(0, np.nan)).quantile(0.95)
            )
        ).fillna(False),
        "All amenity flags are zero": context["is_amenities_listed"].eq(0),
        "Sparse locations under 5 training records": context["location"].map(train_location_counts).fillna(0).lt(5),
    }

    for name, mask in scenarios.items():
        mask_array = np.asarray(mask, dtype=bool)
        if mask_array.sum() == 0:
            continue
        row = regression_metrics(y_values[mask_array], predictions[mask_array])
        row.update({"scenario": name, "records": int(mask_array.sum())})
        tests.append(row)
    return pd.DataFrame(tests)


def search_sensitivity_note(search_log: pd.DataFrame, best_params: dict) -> str:
    depth_col = "param_model__regressor__max_depth"
    trees_col = "param_model__regressor__n_estimators"
    notes = [f"The best logged trial achieved PKR {search_log['validation_rmse_pkr'].min():,.0f} validation RMSE."]
    if depth_col in search_log:
        depth_summary = search_log.groupby(depth_col)["validation_rmse_pkr"].mean().sort_values()
        notes.append(f"Across tested trials, max_depth={int(depth_summary.index[0])} had the lowest average validation RMSE.")
    if trees_col in search_log:
        tree_summary = search_log.groupby(trees_col)["validation_rmse_pkr"].mean().sort_values()
        notes.append(f"The strongest average estimator count among sampled trials was {int(tree_summary.index[0])} trees.")
    notes.append(f"Selected parameters: {best_params}.")
    return " ".join(notes)


def write_report(context: dict, path: Path) -> None:
    comparison = pd.DataFrame(context["comparison"]).to_markdown(index=False)
    stress = pd.DataFrame(context["stress_tests"]).to_markdown(index=False)
    feature_change = pd.DataFrame(context["feature_selection"]).to_markdown(index=False)
    optimization = pd.DataFrame(context["optimization"]).to_markdown(index=False)
    audit = context["cleaning_audit"]
    text = f"""# Lahore Real Estate Price Prediction - Results Report

## Dataset and cleaning

The project uses the supplied `lahore_house_listings_zameen.csv` file. It contains Lahore house listings collected from Zameen.com. The raw file had **{audit['initial_rows']:,} rows** and the cleaned modeling dataset has **{audit['final_rows']:,} rows**.

Cleaning applied:

- Converted Crore, Lakh and Arab price text to PKR.
- Converted Marla and Kanal area values to square feet using 1 Marla = 225 sq ft and 1 Kanal = 4,500 sq ft.
- Kept residential listings marked for sale and enforced the assignment price/area boundaries.
- Replaced invalid room-count values with missing values for fold-safe median imputation.
- Removed {audit['link_duplicates_removed']:,} repeated links and {audit['near_duplicates_removed']:,} near-duplicate listings using location, area, bedrooms, bathrooms and a 2% price tolerance.
- Extracted safe title indicators such as corner, park-facing, basement, brand-new and double-unit without using raw title text as a model input.

## Validation and tuning

The target is modeled in `log1p(price)` space, but all scoring and reporting are performed after inverse transformation in original PKR. Five-fold price-binned stratified cross-validation was used for both model families. Every XGBoost trial is stored in `tuning/trial_log.csv`, and every Ridge trial is stored in `tuning/ridge_trial_log.csv`.

{context['tuning_note']}

## Model comparison

{comparison}

The selected production model is **{context['selected_model']}**. Model choice considered RMSE, MAE, MAPE, latency, artifact size, interpretability and maintenance burden.

## Prediction interval validation

The project uses split-conformal calibration on absolute log-price residuals. Empirical held-out coverage is **{context['coverage_percent']:.2f}%** for the requested 90% interval. Median interval width on the held-out set is **PKR {context['median_interval_width_pkr']:,.0f}**.

## Feature selection

{feature_change}

SHAP and permutation importance outputs are saved in `results/tables/`. Protected domain features such as location, area, bedrooms and bathrooms are not removed automatically. The trimmed model is eligible only when RMSE deterioration remains below 3%, and it is selected only when it also provides a practical accuracy, size or latency benefit.

## Stress scenarios

{stress}

## Deployment optimization

{optimization}

The complete preprocessor and final model are serialized together in `artifacts/final_pipeline.joblib`. `deployment/load_and_predict.py` returns `price_pkr`, `lower_90` and `upper_90` from a raw listing dictionary.

## Leakage re-audit

No training feature is calculated from the target price. `price_per_sqft` is deliberately excluded because it would use the answer being predicted. Raw listing titles and URLs are also excluded; only predefined keyword flags that are available before sale are retained.
"""
    path.write_text(text, encoding="utf-8")
