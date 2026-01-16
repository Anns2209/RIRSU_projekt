from __future__ import annotations

import os
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import joblib
from flask import Flask, request, jsonify
from tensorflow import keras

# ------------------------------------------------------------
# REST API za napoved PM10 (časovne vrste + GRU/LSTM)
# ------------------------------------------------------------

APP_DIR = os.path.dirname(os.path.abspath(__file__))
ART_DIR = os.path.join(APP_DIR, "artifacts")

MODEL_PATH = os.path.join(ART_DIR, "pm10_gru_model.keras")
PREPROCESSOR_PATH = os.path.join(ART_DIR, "preprocessor.joblib")
CONFIG_PATH = os.path.join(ART_DIR, "config.joblib")

app = Flask(__name__)

# Naložimo artefakte ob zagonu strežnika 
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model not found: {MODEL_PATH}")
if not os.path.exists(PREPROCESSOR_PATH):
    raise FileNotFoundError(f"Preprocessor not found: {PREPROCESSOR_PATH}")
if not os.path.exists(CONFIG_PATH):
    raise FileNotFoundError(f"Config not found: {CONFIG_PATH}")

model = keras.models.load_model(MODEL_PATH)
preprocessor = joblib.load(PREPROCESSOR_PATH)
config = joblib.load(CONFIG_PATH)

WINDOW = int(config["window"])

# Featureji, ki jih pričakujemo v input JSON-u
NUMERIC_FEATURES = list(config["numeric_features"])
CATEGORICAL_FEATURES = list(config["categorical_features"])
INPUT_COLS = NUMERIC_FEATURES + CATEGORICAL_FEATURES


def _to_dense(x):
    """Keras lažje dela z dense; ColumnTransformer lahko vrne sparse."""
    return x.toarray() if hasattr(x, "toarray") else x


def _validate_payload(payload: Any) -> List[Dict[str, Any]]:
    """Validacija JSON telesa zahtevka."""
    if payload is None or not isinstance(payload, dict):
        raise ValueError("Missing JSON body or invalid JSON object.")

    if "data" not in payload:
        raise ValueError("Missing 'data' field in JSON body.")

    data = payload["data"]
    if not isinstance(data, list):
        raise ValueError("'data' must be a list of objects (rows).")

    if len(data) != WINDOW:
        raise ValueError(f"'data' must have exactly {WINDOW} rows, got {len(data)}.")

    for i, row in enumerate(data):
        if not isinstance(row, dict):
            raise ValueError(f"Row {i} in 'data' is not an object/dict.")

    return data


@app.route("/health", methods=["GET"])
def health():
    """Enostaven health check (uporabno za Docker/compose)."""
    return jsonify({"status": "ok", "window": WINDOW})


@app.route("/predict", methods=["POST"])
def predict():
 
    try:
        payload = request.get_json(silent=True)
        data_rows = _validate_payload(payload)

        df_in = pd.DataFrame(data_rows)

        # Preverimo manjkajoče stolpce
        missing_cols = [c for c in INPUT_COLS if c not in df_in.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")

        # stolpce v pravem vrstnem redu
        df_in = df_in[INPUT_COLS].copy()

        # Transformacija
        X = preprocessor.transform(df_in)
        X = _to_dense(X)

        # Oblika za RNN
        X_seq = X.reshape(1, WINDOW, X.shape[1])

        # Napoved 
        pred_log = float(model.predict(X_seq, verbose=0).squeeze())
        pred_pm10 = float(np.expm1(pred_log))

        return jsonify({"prediction": pred_pm10})

    except Exception as e:

        return jsonify({"error": str(e)}), 400


if __name__ == "__main__":
    # zagon
    app.run(host="0.0.0.0", port=5005, debug=True)
