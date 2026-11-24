import json
from pathlib import Path
import pandas as pd
import xgboost as xgb

# Project root directory
ROOT_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = ROOT_DIR / "models"

# Load metadata
with open(MODEL_DIR / "metadata.json", "r") as file:
    metadata = json.load(file)

FEATURES = metadata["features"]
THRESHOLDS = metadata["thresholds"]

# Load all trained models
MODELS = {}

for horizon in metadata["forecast_horizons"]:
    model = xgb.XGBClassifier()
    model.load_model(str(MODEL_DIR / f"xgboost_{horizon}day.json"))
    MODELS[horizon] = model


def predict_unrest(features: dict, horizon: int = 1):
    """
    Predict the probability of civil unrest occurring
    within the selected number of days.
    """

    if horizon not in MODELS:
        raise ValueError("Horizon must be 1, 3, or 7 days.")

    missing_features = [f for f in FEATURES if f not in features]

    if missing_features:
        raise ValueError(f"Missing features: {missing_features}")

    # Arrange features in the exact order used during training
    input_data = pd.DataFrame(
        [[features[f] for f in FEATURES]],
        columns=FEATURES
    )

    # Get prediction probability
    probability = float(MODELS[horizon].predict_proba(input_data)[0][1])

    # Apply the validation-selected threshold
    threshold = float(THRESHOLDS[str(horizon)])

    prediction = int(probability >= threshold)

    return {
        "horizon_days": horizon,
        "probability": round(probability, 4),
        "threshold": threshold,
        "prediction": prediction,
        "forecast": (
            "Unrest event expected"
            if prediction == 1
            else "No unrest event expected"
        )
    }