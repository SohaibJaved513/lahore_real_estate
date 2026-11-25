import pandas as pd

from realestate.data import canonicalize, parse_area, parse_price


def test_unit_parsers():
    assert parse_price("2.5 Crore") == 25_000_000
    assert parse_price("75 Lakh") == 7_500_000
    assert parse_area("10 Marla") == 2250
    assert parse_area("1 Kanal") == 4500


def test_dataset_mapping():
    frame = pd.DataFrame([{
        "Title": "Brand New Corner House",
        "Location": "DHA Phase 6, DHA Defence, Lahore, Punjab",
        "Type": "House",
        "Price": "5 Crore",
        "Area": "10 Marla",
        "Bedrooms": "5",
        "Bathrooms": "6",
        "Purpose": "For Sale",
        "Built Year": 2024,
        "Kitchens": 2,
        "Store Rooms": 1,
        "Servant Quarters": 1,
        "Furnished": True,
        "Gym": False,
        "Study Room": True,
        "Drawing Room": True,
        "Dining Room": True,
        "Lawn/Garden": True,
        "Swimming Pool": False,
        "Electricity Backup": True,
        "Lounge/Sitting Room": True,
        "Link": "sample",
    }])
    result, _ = canonicalize(frame)
    assert result.loc[0, "area_sqft"] == 2250
    assert result.loc[0, "price"] == 50_000_000
    assert result.loc[0, "corner"] == 1
    assert result.loc[0, "brand_new"] == 1
