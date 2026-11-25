import pandas as pd

from realestate.features import MODEL_FEATURES, make_features


def test_feature_builder_returns_expected_columns():
    frame = pd.DataFrame([{
        "location": "DHA Phase 6, DHA Defence",
        "city_area": "DHA Defence",
        "property_type": "House",
        "area_sqft": 2250,
        "bedrooms": 5,
        "bathrooms": 6,
        "kitchens": 2,
        "store_rooms": 1,
        "servant_quarters": 1,
        "property_age": 1,
        "location_depth": 2,
        "drawing_room": 1,
        "dining_room": 1,
        "lawn_garden": 1,
        "electricity_backup": 1,
        "lounge_sitting_room": 1,
    }])
    result = make_features(frame)
    assert set(MODEL_FEATURES).issubset(result.columns)
    assert result.loc[0, "society_tier"] == "Premium"
    assert result.loc[0, "amenities_count"] == 5
