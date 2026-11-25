from realestate.config import FEATURES_PATH
from realestate.data import load_property_data
from realestate.features import make_features
from realestate.metrics import deployment_latency
from deployment.load_and_predict import load_model
import json

if __name__ == "__main__":
    frame, _, _ = load_property_data()
    data = make_features(frame)
    features = json.loads(FEATURES_PATH.read_text(encoding="utf-8"))["model_features"]
    sample = data[features].head(100)
    print(deployment_latency(load_model(), sample.iloc[[0]], sample))
