# Deployment Optimization Report

| version                   |   artifact_size_mb |   single_prediction_ms |   batch_100_ms |    rmse_pkr |
|:--------------------------|-------------------:|-----------------------:|---------------:|------------:|
| Full XGBoost tuned        |           0.315783 |                4.76607 |        5.69949 | 2.41672e+07 |
| Trimmed XGBoost tuned     |           0.302086 |                4.72247 |        5.42283 | 2.47663e+07 |
| Selected production model |           0.315783 |                4.76607 |        5.69949 | 2.41672e+07 |

The feature-trimmed model is used only when held-out RMSE remains within the 3% tolerance. Single-listing latency is the median across 1,000 runs; batch latency is the median across 100 runs.
