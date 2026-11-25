# Lahore Real Estate Price Prediction - Results Report

## Dataset and cleaning

The project uses the supplied `lahore_house_listings_zameen.csv` file. It contains Lahore house listings collected from Zameen.com. The raw file had **18,249 rows** and the cleaned modeling dataset has **16,575 rows**.

Cleaning applied:

- Converted Crore, Lakh and Arab price text to PKR.
- Converted Marla and Kanal area values to square feet using 1 Marla = 225 sq ft and 1 Kanal = 4,500 sq ft.
- Kept residential listings marked for sale and enforced the assignment price/area boundaries.
- Replaced invalid room-count values with missing values for fold-safe median imputation.
- Removed 0 repeated links and 1,582 near-duplicate listings using location, area, bedrooms, bathrooms and a 2% price tolerance.
- Extracted safe title indicators such as corner, park-facing, basement, brand-new and double-unit without using raw title text as a model input.

## Validation and tuning

The target is modeled in `log1p(price)` space, but all scoring and reporting are performed after inverse transformation in original PKR. Five-fold price-binned stratified cross-validation was used for both model families. Every XGBoost trial is stored in `tuning/trial_log.csv`, and every Ridge trial is stored in `tuning/ridge_trial_log.csv`.

The best logged trial achieved PKR 24,900,841 validation RMSE. Across tested trials, max_depth=6 had the lowest average validation RMSE. The strongest average estimator count among sampled trials was 320 trees. Selected parameters: {'model__regressor__subsample': 0.75, 'model__regressor__reg_lambda': 1.2, 'model__regressor__reg_alpha': 0.2, 'model__regressor__n_estimators': 320, 'model__regressor__min_child_weight': 2, 'model__regressor__max_depth': 6, 'model__regressor__learning_rate': 0.08, 'model__regressor__colsample_bytree': 0.9}.

## Model comparison

|    rmse_pkr |   rmse_percent_mean |   mape_percent |     mae_pkr | model           |   train_time_seconds |   latency_ms_per_prediction | final           |
|------------:|--------------------:|---------------:|------------:|:----------------|---------------------:|----------------------------:|:----------------|
| 2.6421e+07  |             46.7808 |        16.8313 | 1.15257e+07 | XGBoost untuned |                1.606 |                       5.012 | No              |
| 2.41672e+07 |             42.7902 |        15.3618 | 1.04745e+07 | XGBoost tuned   |                2.008 |                       5.156 | Candidate       |
| 1.18338e+08 |            209.529  |        52.8574 | 2.8411e+07  | Ridge tuned     |                0.89  |                       4.902 | No              |
| 3.0336e+07  |             53.7126 |        18.187  | 1.1592e+07  | Weighted blend  |                2.898 |                      10.058 | Comparison only |

The selected production model is **XGBoost tuned**. Model choice considered RMSE, MAE, MAPE, latency, artifact size, interpretability and maintenance burden.

## Prediction interval validation

The project uses split-conformal calibration on absolute log-price residuals. Empirical held-out coverage is **92.01%** for the requested 90% interval. Median interval width on the held-out set is **PKR 25,636,294**.

## Feature selection

| version             |   feature_count | removed_features                                                                               |    rmse_pkr |   rmse_change_percent |
|:--------------------|----------------:|:-----------------------------------------------------------------------------------------------|------------:|----------------------:|
| Full feature set    |              34 | None                                                                                           | 2.41672e+07 |               0       |
| Trimmed feature set |              28 | servant_quarters, kitchens, electricity_backup, study_room, lounge_sitting_room, property_type | 2.47663e+07 |               2.47921 |

SHAP and permutation importance outputs are saved in `results/tables/`. Protected domain features such as location, area, bedrooms and bathrooms are not removed automatically. The trimmed model is eligible only when RMSE deterioration remains below 3%, and it is selected only when it also provides a practical accuracy, size or latency benefit.

## Stress scenarios

|    rmse_pkr |   rmse_percent_mean |   mape_percent |     mae_pkr | scenario                                  |   records |
|------------:|--------------------:|---------------:|------------:|:------------------------------------------|----------:|
| 1.50831e+08 |             34.3158 |        31.4583 | 1.38411e+08 | Top 1% price listings                     |        35 |
| 5.79813e+07 |             43.8012 |        18.9334 | 3.1225e+07  | Unusual area-to-bedroom ratio             |       300 |
| 1.72007e+07 |             43.0919 |        18.0181 | 8.13755e+06 | All amenity flags are zero                |       178 |
| 1.94299e+07 |             42.3012 |        20.7542 | 9.38705e+06 | Sparse locations under 5 training records |       467 |

## Deployment optimization

| version                   |   artifact_size_mb |   single_prediction_ms |   batch_100_ms |    rmse_pkr |
|:--------------------------|-------------------:|-----------------------:|---------------:|------------:|
| Full XGBoost tuned        |           0.315783 |                4.76607 |        5.69949 | 2.41672e+07 |
| Trimmed XGBoost tuned     |           0.302086 |                4.72247 |        5.42283 | 2.47663e+07 |
| Selected production model |           0.315783 |                4.76607 |        5.69949 | 2.41672e+07 |

The complete preprocessor and final model are serialized together in `artifacts/final_pipeline.joblib`. `deployment/load_and_predict.py` returns `price_pkr`, `lower_90` and `upper_90` from a raw listing dictionary.

## Leakage re-audit

No training feature is calculated from the target price. `price_per_sqft` is deliberately excluded because it would use the answer being predicted. Raw listing titles and URLs are also excluded; only predefined keyword flags that are available before sale are retained.
