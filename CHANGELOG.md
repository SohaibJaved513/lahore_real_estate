# Changelog

## Week 2 - Dataset-specific completion

- Replaced demonstration data with the supplied 18,249-row Lahore Zameen.com CSV.
- Added exact parsing for Crore, Lakh, Arab, Marla and Kanal values.
- Added dataset-specific room, build-year, amenity and title-feature cleaning.
- Added five-fold XGBoost and Ridge optimization logs.
- Added SHAP, permutation importance and amenity correlation analysis.
- Added split-conformal log-price prediction intervals.
- Added four quantified stress scenarios.
- Added complete model serialization and deployment latency profiling.
- Rebuilt the Flask interface around the dataset's house-only scope.
