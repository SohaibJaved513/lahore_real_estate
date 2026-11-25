from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

from realestate.config import (
    AREA_MAX_SQFT,
    AREA_MIN_SQFT,
    PRICE_MAX,
    PRICE_MIN,
    REAL_DATA_PATH,
    REFERENCE_YEAR,
)

ALIASES = {
    "title": ["title", "property_title"],
    "location_raw": ["location", "address", "locality"],
    "property_type": ["type", "property_type", "property type"],
    "price": ["price", "property_price", "amount"],
    "area": ["area", "area_size", "size"],
    "bedrooms": ["bedrooms", "beds", "bedroom"],
    "bathrooms": ["bathrooms", "baths", "bathroom"],
    "purpose": ["purpose", "listing_type"],
    "date_posted": ["date posted", "date_posted", "date_added"],
    "built_year": ["built year", "built_year", "year_built"],
    "kitchens": ["kitchens", "kitchen"],
    "store_rooms": ["store rooms", "store_rooms", "storerooms"],
    "servant_quarters": ["servant quarters", "servant_quarters"],
    "source_link": ["link", "url", "source_link"],
}

BOOLEAN_ALIASES = {
    "furnished": ["furnished"],
    "gym": ["gym"],
    "study_room": ["study room", "study_room"],
    "drawing_room": ["drawing room", "drawing_room"],
    "dining_room": ["dining room", "dining_room"],
    "lawn_garden": ["lawn/garden", "lawn garden", "lawn_garden"],
    "swimming_pool": ["swimming pool", "swimming_pool"],
    "electricity_backup": ["electricity backup", "electricity_backup"],
    "lounge_sitting_room": ["lounge/sitting room", "lounge_sitting_room"],
}

TITLE_PATTERNS = {
    "corner": r"\bcorner\b",
    "park_facing": r"\bpark[ -]?facing\b|\bfacing park\b",
    "basement": r"\bbasement\b",
    "brand_new": r"\bbrand new\b|\bnewly built\b",
    "double_unit": r"\bdouble unit\b",
    "under_construction": r"\bunder construction\b|\binstallment plan\b|\binstalment plan\b",
}


def clean_name(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(value).strip().lower()).strip("_")


def find_column(columns: list[str], aliases: list[str]) -> str | None:
    normalized = {clean_name(column): column for column in columns}
    for alias in aliases:
        match = normalized.get(clean_name(alias))
        if match:
            return match
    return None


def parse_price(value: object) -> float:
    if pd.isna(value):
        return np.nan
    if isinstance(value, (int, float, np.number)):
        return float(value)
    text = str(value).lower().replace(",", "").replace("pkr", "").strip()
    match = re.search(r"[-+]?\d*\.?\d+", text)
    if not match:
        return np.nan
    number = float(match.group())
    if "arab" in text or "billion" in text:
        number *= 1_000_000_000
    elif "crore" in text or re.search(r"\bcr\b", text):
        number *= 10_000_000
    elif "lakh" in text or "lac" in text:
        number *= 100_000
    elif "million" in text:
        number *= 1_000_000
    elif "thousand" in text:
        number *= 1_000
    return number


def parse_area(value: object) -> float:
    if pd.isna(value):
        return np.nan
    text = str(value).lower().replace(",", " ")
    match = re.search(r"[-+]?\d*\.?\d+", text)
    if not match:
        return np.nan
    number = float(match.group())
    if "kanal" in text:
        return number * 4_500
    if "marla" in text:
        return number * 225
    if "yard" in text:
        return number * 9
    if "meter" in text or "sqm" in text:
        return number * 10.7639
    return number


def to_numeric_count(value: object, upper: int = 15) -> float:
    if pd.isna(value) or str(value).strip() in {"", "-", "nan", "None"}:
        return np.nan
    match = re.search(r"\d+", str(value))
    if not match:
        return np.nan
    number = int(match.group())
    return float(number) if 0 <= number <= upper else np.nan


def to_binary(value: object) -> int:
    if pd.isna(value):
        return 0
    if isinstance(value, (bool, np.bool_)):
        return int(value)
    if isinstance(value, (int, float, np.number)):
        return int(float(value) > 0)
    return int(str(value).strip().lower() in {"1", "yes", "true", "y", "available"})


def location_parts(value: object) -> list[str]:
    parts = [part.strip() for part in str(value).split(",") if part.strip()]
    excluded = {"lahore", "punjab", "pakistan"}
    return [part for part in parts if part.lower() not in excluded]


def normalized_location(value: object) -> str:
    parts = location_parts(value)
    return ", ".join(parts) if parts else "Unknown"


def broad_area(value: object) -> str:
    parts = location_parts(value)
    return parts[-1] if parts else "Unknown"


def map_frame(frame: pd.DataFrame) -> pd.DataFrame:
    result = pd.DataFrame(index=frame.index)
    columns = list(frame.columns)
    for canonical, aliases in ALIASES.items():
        source = find_column(columns, aliases)
        if source:
            result[canonical] = frame[source]
    for canonical, aliases in BOOLEAN_ALIASES.items():
        source = find_column(columns, aliases)
        result[canonical] = frame[source].map(to_binary) if source else 0
    return result


def drop_near_duplicates(data: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    keys = [
        "location", "area_sqft", "bedrooms", "bathrooms", "built_year", "kitchens",
        "furnished", "gym", "study_room", "drawing_room", "dining_room", "lawn_garden",
        "swimming_pool", "electricity_backup", "lounge_sitting_room",
    ]
    ordered = data.sort_values(keys + ["price"]).copy()
    previous = ordered.groupby(keys, dropna=False)["price"].shift()
    relative_gap = (ordered["price"] - previous).abs() / previous.replace(0, np.nan)
    duplicate_mask = previous.notna() & relative_gap.le(0.02)
    return ordered.loc[~duplicate_mask].sort_index(), int(duplicate_mask.sum())


def canonicalize(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int | float | str]]:
    data = map_frame(frame)
    required = ["price", "area", "location_raw"]
    missing = [column for column in required if column not in data]
    if missing:
        raise ValueError(f"Required dataset columns were not found: {', '.join(missing)}")

    initial_rows = len(data)
    data["price"] = data["price"].map(parse_price)
    data["area_sqft"] = data["area"].map(parse_area)
    data["location"] = data["location_raw"].map(normalized_location)
    data["city_area"] = data["location_raw"].map(broad_area)
    data["location_depth"] = data["location_raw"].map(lambda value: len(location_parts(value)))

    defaults = {
        "title": "",
        "property_type": "House",
        "bedrooms": np.nan,
        "bathrooms": np.nan,
        "purpose": "For Sale",
        "date_posted": pd.NaT,
        "built_year": np.nan,
        "kitchens": np.nan,
        "store_rooms": np.nan,
        "servant_quarters": np.nan,
        "source_link": "",
    }
    for column, default in defaults.items():
        if column not in data:
            data[column] = default

    data["title"] = data["title"].fillna("").astype(str)
    data["property_type"] = data["property_type"].fillna("House").astype(str).str.strip().str.title()
    data["purpose"] = data["purpose"].fillna("For Sale").astype(str).str.strip().str.title()
    data["date_posted"] = pd.to_datetime(data["date_posted"], errors="coerce")
    data["built_year"] = pd.to_numeric(data["built_year"], errors="coerce")
    data.loc[~data["built_year"].between(1900, REFERENCE_YEAR + 5), "built_year"] = np.nan
    data["built_year_missing"] = data["built_year"].isna().astype(int)
    data["property_age"] = (REFERENCE_YEAR - data["built_year"]).clip(lower=0, upper=100)

    for column in ["bedrooms", "bathrooms", "kitchens", "store_rooms", "servant_quarters"]:
        upper = 20 if column in {"bedrooms", "bathrooms"} else 10
        data[column] = data[column].map(lambda value, limit=upper: to_numeric_count(value, limit))

    title_lower = data["title"].str.lower()
    for feature, pattern in TITLE_PATTERNS.items():
        data[feature] = title_lower.str.contains(pattern, regex=True, na=False).astype(int)
    data["under_construction"] = (
        (data["under_construction"] == 1) | (data["built_year"] > REFERENCE_YEAR)
    ).astype(int)

    sale_mask = data["purpose"].str.contains("sale", case=False, na=False)
    residential_mask = ~data["property_type"].str.contains("commercial|office|shop", case=False, na=False)
    price_mask = data["price"].between(PRICE_MIN, PRICE_MAX)
    area_mask = data["area_sqft"].between(AREA_MIN_SQFT, AREA_MAX_SQFT)
    valid_core = data["location"].ne("Unknown") & data["price"].notna() & data["area_sqft"].notna()

    data = data.loc[sale_mask & residential_mask & price_mask & area_mask & valid_core].copy()
    after_filters = len(data)
    link_duplicates = int(data["source_link"].ne("").mul(data["source_link"].duplicated()).sum())
    if "source_link" in data:
        data = data.loc[~(data["source_link"].ne("") & data["source_link"].duplicated())]
    data, near_duplicates = drop_near_duplicates(data)

    keep_columns = [
        "title", "location", "city_area", "location_depth", "property_type", "price", "area_sqft",
        "bedrooms", "bathrooms", "purpose", "date_posted", "built_year", "built_year_missing",
        "property_age", "kitchens", "store_rooms", "servant_quarters", "furnished", "gym",
        "study_room", "drawing_room", "dining_room", "lawn_garden", "swimming_pool",
        "electricity_backup", "lounge_sitting_room", "corner", "park_facing", "basement",
        "brand_new", "double_unit", "under_construction", "source_link",
    ]
    data = data[keep_columns].reset_index(drop=True)
    audit = {
        "source_file": REAL_DATA_PATH.name,
        "initial_rows": initial_rows,
        "rows_after_required_filters": after_filters,
        "link_duplicates_removed": link_duplicates,
        "near_duplicates_removed": near_duplicates,
        "final_rows": len(data),
        "rows_removed_total": initial_rows - len(data),
        "price_min_pkr": float(data["price"].min()),
        "price_max_pkr": float(data["price"].max()),
        "median_price_pkr": float(data["price"].median()),
    }
    return data, audit


def load_property_data(path: str | Path | None = None) -> tuple[pd.DataFrame, str, dict]:
    selected = Path(path) if path else REAL_DATA_PATH
    if not selected.exists():
        raise FileNotFoundError(f"Dataset not found: {selected}")
    frame = pd.read_csv(selected, low_memory=False)
    clean, audit = canonicalize(frame)
    return clean, "lahore_house_listings_zameen", audit
