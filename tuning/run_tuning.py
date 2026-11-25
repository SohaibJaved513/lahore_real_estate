from realestate.config import TUNING_DIR
from realestate.data import load_property_data
from realestate.features import MODEL_FEATURES, make_features
from realestate.modeling import tune_ridge, tune_xgb

if __name__ == "__main__":
    frame, _, _ = load_property_data()
    data = make_features(frame)
    X, y = data[MODEL_FEATURES], data["price"]
    _, _, xgb = tune_xgb(X, y, TUNING_DIR / "trial_log.csv")
    _, _, ridge = tune_ridge(X, y, TUNING_DIR / "ridge_trial_log.csv")
    print({"xgboost": xgb, "ridge": ridge})
