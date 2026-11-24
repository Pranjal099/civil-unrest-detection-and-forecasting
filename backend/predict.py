import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Mapping, Union

import numpy as np
import pandas as pd
import xgboost as xgb

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Project root and model directory
ROOT_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = ROOT_DIR / "models"

# Load metadata
metadata_path = MODEL_DIR / "metadata.json"
if not metadata_path.exists():
    raise FileNotFoundError(f"Model metadata not found at {metadata_path}")

with open(metadata_path, "r", encoding="utf-8") as file:
    metadata = json.load(file)

FEATURES: List[str] = metadata["features"]
THRESHOLDS: Dict[str, float] = {str(k): float(v) for k, v in metadata["thresholds"].items()}
FORECAST_HORIZONS: List[int] = metadata["forecast_horizons"]

# Load all trained XGBoost models
MODELS: Dict[int, xgb.XGBClassifier] = {}

for horizon in FORECAST_HORIZONS:
    model_file = MODEL_DIR / f"xgboost_{horizon}day.json"
    if not model_file.exists():
        raise FileNotFoundError(f"Model file not found: {model_file}")
    model = xgb.XGBClassifier()
    model.load_model(str(model_file))
    MODELS[horizon] = model


def validate_feature_payload(features: Mapping[str, Any]) -> Dict[str, float]:
    """
    Validate and sanitize input feature dictionary.

    Raises
    ------
    TypeError
        If features is not a dictionary or contains non-numeric values.
    ValueError
        If required features are missing or contain NaN/negative values.
    """
    if not isinstance(features, (dict, Mapping)):
        raise TypeError(f"Features must be a dictionary or mapping, got {type(features).__name__}.")

    missing = [f for f in FEATURES if f not in features]
    if missing:
        raise ValueError(f"Missing required features: {missing}")

    validated = {}
    for f in FEATURES:
        val = features[f]
        if val is None or (isinstance(val, float) and np.isnan(val)):
            raise ValueError(f"Feature '{f}' contains NaN or None value.")
        try:
            num_val = float(val)
        except (ValueError, TypeError) as e:
            raise TypeError(f"Feature '{f}' must be numeric, got {val!r}") from e

        if num_val < 0:
            raise ValueError(f"Feature '{f}' cannot be negative, got {num_val}")
        validated[f] = num_val

    return validated


def predict_unrest(features: Mapping[str, Any], horizon: int = 1) -> Dict[str, Any]:
    """
    Predict the probability of civil unrest occurring within a selected forecast horizon.

    Parameters
    ----------
    features : dict
        Mapping of historical activity features (lag_1, lag_3, lag_7, lag_14,
        rolling_7_mean, rolling_7_sum, rolling_14_mean).
    horizon : int, default 1
        Forecast window in days (must be 1, 3, or 7).

    Returns
    -------
    dict
        Structured forecast with horizon_days, probability, threshold,
        binary prediction, and human-readable forecast message.
    """
    if horizon not in MODELS:
        raise ValueError(f"Invalid horizon {horizon}. Supported horizons: {list(MODELS.keys())}.")

    valid_features = validate_feature_payload(features)

    # Arrange features in the exact column order used during training
    input_data = pd.DataFrame(
        [[valid_features[f] for f in FEATURES]],
        columns=FEATURES
    )

    # Inference probability
    probability = float(MODELS[horizon].predict_proba(input_data)[0][1])

    # Calibrated decision threshold from validation tuning
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


def predict_unrest_batch(
    features_df: pd.DataFrame,
    horizon: int = 1
) -> pd.DataFrame:
    """
    Execute batch civil unrest inference across multiple time-series records.

    Parameters
    ----------
    features_df : pd.DataFrame
        DataFrame containing all required FEATURES.
    horizon : int, default 1
        Forecast horizon in days (1, 3, or 7).

    Returns
    -------
    pd.DataFrame
        Copy of features_df with added 'probability', 'prediction', and 'forecast' columns.
    """
    if horizon not in MODELS:
        raise ValueError(f"Invalid horizon {horizon}. Supported horizons: {list(MODELS.keys())}.")

    missing = [f for f in FEATURES if f not in features_df.columns]
    if missing:
        raise ValueError(f"Missing required feature columns: {missing}")

    X = features_df[FEATURES]
    probs = MODELS[horizon].predict_proba(X)[:, 1]
    threshold = float(THRESHOLDS[str(horizon)])
    preds = (probs >= threshold).astype(int)

    result_df = features_df.copy()
    result_df["probability"] = np.round(probs, 4)
    result_df["threshold"] = threshold
    result_df["prediction"] = preds
    result_df["forecast"] = np.where(preds == 1, "Unrest event expected", "No unrest event expected")
    return result_df
