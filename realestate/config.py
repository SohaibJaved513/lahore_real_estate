from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
ARTIFACT_DIR = ROOT / "artifacts"
RESULTS_DIR = ROOT / "results"
PLOTS_DIR = RESULTS_DIR / "plots"
TABLES_DIR = RESULTS_DIR / "tables"
TUNING_DIR = ROOT / "tuning"

REAL_DATA_PATH = RAW_DIR / "lahore_house_listings_zameen.csv"
PROCESSED_DATA_PATH = PROCESSED_DIR / "lahore_house_listings_clean.csv"
MODEL_PATH = ARTIFACT_DIR / "final_pipeline.joblib"
INTERVAL_PATH = ARTIFACT_DIR / "interval_calibration.json"
METRICS_PATH = ARTIFACT_DIR / "model_metrics.json"
FEATURES_PATH = ARTIFACT_DIR / "feature_schema.json"
UI_OPTIONS_PATH = ARTIFACT_DIR / "ui_options.json"
CLEANING_AUDIT_PATH = ARTIFACT_DIR / "cleaning_audit.json"

RANDOM_STATE = 42
TEST_SIZE = 0.20
CALIBRATION_SIZE = 0.20
REFERENCE_YEAR = 2025
PRICE_MIN = 1_000_000
PRICE_MAX = 500_000_000
AREA_MIN_SQFT = 225
AREA_MAX_SQFT = 45_000
