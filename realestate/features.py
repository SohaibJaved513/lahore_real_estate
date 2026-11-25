from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, TargetEncoder

AMENITY_FEATURES = [
    "furnished",
    "gym",
    "study_room",
    "drawing_room",
    "dining_room",
    "lawn_garden",
    "swimming_pool",
    "electricity_backup",
    "lounge_sitting_room",
]

NUMERIC_FEATURES = [
    "area_sqft",
    "area_log",
    "bedrooms",
    "bathrooms",
    "kitchens",
    "store_rooms",
    "servant_quarters",
    "property_age",
    "built_year_missing",
    "location_depth",
    "bed_bath_interaction",
    "bedroom_bath_ratio",
    "area_per_bedroom",
    "amenities_count",
    "is_amenities_listed",
    *AMENITY_FEATURES,
    "corner",
    "park_facing",
    "basement",
    "brand_new",
    "double_unit",
    "under_construction",
]
HIGH_CARDINALITY = ["location"]
LOW_CARDINALITY = ["city_area", "property_type", "society_tier"]
MODEL_FEATURES = HIGH_CARDINALITY + LOW_CARDINALITY + NUMERIC_FEATURES
PROTECTED_FEATURES = {"location", "city_area", "area_sqft", "area_log", "bedrooms", "bathrooms"}


def society_tier(location: str) -> str:
    text = str(location).lower()
    premium = ["dha", "defence", "bahria", "cantt"]
    mid = [
        "johar", "gulberg", "model town", "wapda", "valencia", "iqbal town",
        "garden town", "faisal town", "lake city", "park view", "paragon",
    ]
    if any(name in text for name in premium):
        return "Premium"
    if any(name in text for name in mid):
        return "Mid"
    return "Standard"


def make_features(frame: pd.DataFrame) -> pd.DataFrame:
    data = frame.copy()
    numeric_defaults = {
        "area_sqft": np.nan,
        "bedrooms": np.nan,
        "bathrooms": np.nan,
        "kitchens": np.nan,
        "store_rooms": np.nan,
        "servant_quarters": np.nan,
        "property_age": np.nan,
        "built_year_missing": 1,
        "location_depth": 1,
        **{feature: 0 for feature in AMENITY_FEATURES},
        "corner": 0,
        "park_facing": 0,
        "basement": 0,
        "brand_new": 0,
        "double_unit": 0,
        "under_construction": 0,
    }
    text_defaults = {
        "location": "Unknown",
        "city_area": "Unknown",
        "property_type": "House",
    }
    for column, default in {**numeric_defaults, **text_defaults}.items():
        if column not in data:
            data[column] = default

    for column in numeric_defaults:
        data[column] = pd.to_numeric(data[column], errors="coerce")
    data["area_sqft"] = data["area_sqft"].clip(225, 45_000)
    data["bedrooms"] = data["bedrooms"].clip(0, 20)
    data["bathrooms"] = data["bathrooms"].clip(0, 20)
    for column in ["kitchens", "store_rooms", "servant_quarters"]:
        data[column] = data[column].clip(0, 10)
    data["property_age"] = data["property_age"].clip(0, 100)

    data["area_log"] = np.log1p(data["area_sqft"])
    data["bed_bath_interaction"] = data["bedrooms"] * data["bathrooms"]
    data["bedroom_bath_ratio"] = data["bedrooms"] / data["bathrooms"].replace(0, np.nan)
    data["area_per_bedroom"] = data["area_sqft"] / data["bedrooms"].replace(0, np.nan)
    data["amenities_count"] = data[AMENITY_FEATURES].fillna(0).sum(axis=1)
    data["is_amenities_listed"] = data["amenities_count"].gt(0).astype(int)
    data["society_tier"] = (data["city_area"].fillna("") + " " + data["location"].fillna("")).map(society_tier)

    for column in MODEL_FEATURES:
        if column not in data:
            data[column] = 0 if column in NUMERIC_FEATURES else "Unknown"
    return data


def build_preprocessor(feature_columns: list[str] | None = None) -> ColumnTransformer:
    selected = feature_columns or MODEL_FEATURES
    numeric = [column for column in NUMERIC_FEATURES if column in selected]
    high = [column for column in HIGH_CARDINALITY if column in selected]
    low = [column for column in LOW_CARDINALITY if column in selected]

    numeric_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
        ("scaler", StandardScaler()),
    ])
    low_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    high_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", TargetEncoder(target_type="continuous", cv=5, random_state=42, smooth="auto")),
    ])

    transformers = []
    if numeric:
        transformers.append(("numeric", numeric_pipe, numeric))
    if low:
        transformers.append(("low_card", low_pipe, low))
    if high:
        transformers.append(("high_card", high_pipe, high))
    return ColumnTransformer(transformers, remainder="drop", verbose_feature_names_out=True)
