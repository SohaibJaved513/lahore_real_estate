from __future__ import annotations

import json

from flask import Blueprint, jsonify, render_template, request

from deployment.load_and_predict import load_model, predict_price
from realestate.config import METRICS_PATH, MODEL_PATH, UI_OPTIONS_PATH

main = Blueprint("main", __name__)
_model = None


def read_json(path) -> dict:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def get_model():
    global _model
    if _model is None and MODEL_PATH.exists():
        _model = load_model()
    return _model


@main.get("/")
def home():
    return render_template(
        "index.html",
        metrics=read_json(METRICS_PATH),
        options=read_json(UI_OPTIONS_PATH),
        model_ready=MODEL_PATH.exists(),
    )


@main.post("/predict")
def predict():
    model = get_model()
    if model is None:
        return jsonify({"error": "Model artifact is missing. Run python run_pipeline.py first."}), 503
    payload = request.get_json(silent=True) or request.form.to_dict()
    try:
        return jsonify(predict_price(model, payload))
    except (TypeError, ValueError, KeyError) as error:
        return jsonify({"error": f"Invalid property details: {error}"}), 400


@main.get("/health")
def health():
    return jsonify({"status": "ok", "model_ready": MODEL_PATH.exists()})


@main.get("/api/metrics")
def metrics():
    if not METRICS_PATH.exists():
        return jsonify({"error": "Metrics are not available."}), 404
    return jsonify(read_json(METRICS_PATH))
