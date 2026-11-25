# Lahore House Price Prediction Pipeline

A complete machine-learning project built from the supplied `lahore_house_listings_zameen.csv` dataset. The repository converts raw Zameen.com listing fields into a tuned, explainable, uncertainty-aware and deployable Lahore house-price estimator.

## Project scope

The supplied dataset contains **Lahore houses listed for sale**. The model and Flask interface intentionally stay inside that scope; they should not be used for flats, plots, rentals or commercial properties.

## Dataset fields used

The data pipeline reads these source fields:

- `Title`, `Location`, `Type`, `Price`, `Area`, `Bedrooms`, `Bathrooms`, `Purpose`
- `Date Posted`, `Built Year`, `Kitchens`, `Store Rooms`, `Servant Quarters`
- `Furnished`, `Gym`, `Study Room`, `Drawing Room`, `Dining Room`
- `Lawn/Garden`, `Swimming Pool`, `Electricity Backup`, `Lounge/Sitting Room`
- `Link`

`Price` is converted from Crore/Lakh/Arab text to PKR. `Area` is converted from Marla/Kanal to square feet. Invalid room counts become missing values and are imputed inside the fitted pipeline. Safe keyword flags such as corner, park-facing, basement and brand-new are extracted from the title, while the raw title and URL are excluded from training.

## Main requirements completed

- Five-fold price-binned stratified cross-validation
- Logged XGBoost randomized search and Ridge grid search
- Untuned XGBoost, tuned XGBoost, tuned Ridge and weighted-blend comparison
- Original-PKR RMSE, RMSE percentage, MAPE and MAE
- SHAP and permutation importance
- Evidence-based feature trimming with a 3% RMSE tolerance
- Amenity correlation heatmap
- Split-conformal 90% prediction intervals with empirical coverage testing
- Residual analysis by price, area and society
- Stress tests for top-price, unusual-layout, missing-amenity and sparse-location listings
- Complete `joblib` pipeline with preprocessing and model together
- Median single-listing and batch-100 inference latency
- Responsive Flask frontend and JSON prediction endpoint

## Repository structure

```text
app/                 Flask routes, template, CSS and JavaScript
artifacts/           Model, uncertainty calibration, schema and UI options
data/raw/            Supplied Zameen.com CSV
data/processed/      Cleaned and engineered dataset
deployment/          Load/predict interface and benchmark
models/              Model-training entry point
realestate/          Cleaning, features, tuning, evaluation and workflow
results/plots/       Required plots
results/tables/      Comparison and audit tables
tests/               Data, feature and deployment tests
tuning/              Trial logs and best parameters
docs/                Assignment copy, interview notes and LinkedIn draft
```

## Installation

### Windows

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

### macOS or Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run the full pipeline

```bash
python run_pipeline.py
```

For a small development smoke test only:

```bash
python run_pipeline.py --quick
```

The full command regenerates all artifacts, plots, tables and the report using the bundled real dataset.

## Run the Flask application

```bash
python app.py
```

Open `http://127.0.0.1:5000`.

## Prediction interface

```python
from deployment.load_and_predict import load_model, predict_price

model = load_model()
result = predict_price(model, {
    "location": "DHA Phase 6, DHA Defence",
    "city_area": "DHA Defence",
    "area_value": 10,
    "area_unit": "Marla",
    "bedrooms": 5,
    "bathrooms": 6,
    "built_year": 2024,
    "kitchens": 2,
    "store_rooms": 1,
    "servant_quarters": 1,
    "drawing_room": 1,
    "dining_room": 1,
    "lawn_garden": 1,
    "electricity_backup": 1,
    "lounge_sitting_room": 1,
})
```

Returned structure:

```text
{price_pkr, lower_90, upper_90}
```

## Results

The current real-data results are documented in:

- `results/REPORT.md`
- `results/tables/model_comparison.csv`
- `results/tables/feature_selection_comparison.csv`
- `results/tables/stress_scenarios.csv`
- `results/tables/optimization_tradeoff.csv`
- `deployment/OPTIMIZATION_REPORT.md`

## Leakage controls

The target is transformed with `log1p`, but every metric is calculated after inverse transformation in PKR. Target encoding, missing-value imputation and scaling remain inside the scikit-learn pipeline so they are learned separately in every cross-validation fold. `price_per_sqft` is not used as an input because it requires the target price.

## What's new in Week 2

- Reproducible hyperparameter searches with complete trial logs
- Honest comparison across two model families
- SHAP and permutation-based feature evidence
- Prediction uncertainty and coverage validation
- Quantified failure zones and residual patterns
- Production serialization, latency and artifact-size profiling
- A dataset-specific Flask estimator for Lahore houses

Review the report and code before the interview. The grading rubric requires you to explain each decision in your own words.
