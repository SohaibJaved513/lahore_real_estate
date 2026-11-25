from pathlib import Path

import pytest

from deployment.load_and_predict import load_model, predict_price
from realestate.config import MODEL_PATH


@pytest.mark.skipif(not Path(MODEL_PATH).exists(), reason="Model not trained")
def test_prediction_interface():
    output = predict_price(load_model(), {
        "location": "Johar Town",
        "city_area": "Johar Town",
        "area_value": 10,
        "area_unit": "Marla",
        "bedrooms": 5,
        "bathrooms": 5,
        "built_year": 2020,
        "kitchens": 2,
    })
    assert output["lower_90"] <= output["price_pkr"] <= output["upper_90"]
    assert output["price_pkr"] > 0
