from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from realestate.config import FEATURES_PATH, INTERVAL_PATH, MODEL_PATH, REFERENCE_YEAR
from realestate.features import make_features
from realestate.uncertainty import prediction_interval


def load_model(path: str | Path = MODEL_PATH) -> Pipeline:
    return joblib.load(path)


def area_to_sqft(value: float, unit: str) -> float:
    unit_text = str(unit).lower()
    if "kanal" in unit_text:
        return value * 4_500
    if "marla" in unit_text:
        return value * 225
    return value


def prepare_listing(listing: dict) -> pd.DataFrame:
    item = dict(listing)
    if "area_sqft" not in item:
        item["area_sqft"] = area_to_sqft(float(item.get("area_value", 5)), item.get("area_unit", "Marla"))
    built_year = pd.to_numeric(item.get("built_year"), errors="coerce")
    item["built_year_missing"] = int(pd.isna(built_year))
    item["property_age"] = np.nan if pd.isna(built_year) else max(0, REFERENCE_YEAR - float(built_year))
    item["under_construction"] = int(
        float(item.get("under_construction", 0) or 0) > 0
        or (not pd.isna(built_year) and built_year > REFERENCE_YEAR)
    )
    item.setdefault("location", "Unknown")
    item.setdefault("city_area", item["location"].split(",")[-1].strip())
    item.setdefault("property_type", "House")
    item.setdefault("purpose", "For Sale")
    for key in [
        "bedrooms", "bathrooms", "kitchens", "store_rooms", "servant_quarters", "furnished",
        "gym", "study_room", "drawing_room", "dining_room", "lawn_garden", "swimming_pool",
        "electricity_backup", "lounge_sitting_room", "corner", "park_facing", "basement",
        "brand_new", "double_unit",
    ]:
        item[key] = pd.to_numeric(item.get(key, 0), errors="coerce")
    item["location_depth"] = max(1, len([part for part in str(item["location"]).split(",") if part.strip()]))
    return make_features(pd.DataFrame([item]))


def predict_price(model: Pipeline, listing: dict) -> dict[str, float]:
    schema = json.loads(Path(FEATURES_PATH).read_text(encoding="utf-8"))
    features = schema["model_features"]
    prepared = prepare_listing(listing)[features]
    price = float(model.predict(prepared)[0])
    interval = json.loads(Path(INTERVAL_PATH).read_text(encoding="utf-8"))
    lower, upper = prediction_interval([price], float(interval["radius_log"]))
    return {
        "price_pkr": round(price, 2),
        "lower_90": round(float(lower[0]), 2),
        "upper_90": round(float(upper[0]), 2),
    }


if __name__ == "__main__":
    sample = {
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
        "furnished": 1,
        "drawing_room": 1,
        "dining_room": 1,
        "lawn_garden": 1,
        "electricity_backup": 1,
        "lounge_sitting_room": 1,
    }
    print(predict_price(load_model(), sample))
