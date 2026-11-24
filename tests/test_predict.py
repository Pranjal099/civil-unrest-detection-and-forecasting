import pytest
import pandas as pd
import numpy as np

from backend.predict import (
    predict_unrest,
    predict_unrest_batch,
    validate_feature_payload,
    FEATURES,
    MODELS,
    THRESHOLDS
)


def sample_features():
    return {
        "lag_1": 2.0,
        "lag_3": 5.0,
        "lag_7": 3.0,
        "lag_14": 4.0,
        "rolling_7_mean": 1.5,
        "rolling_7_sum": 10.0,
        "rolling_14_mean": 1.2
    }


def test_models_loaded():
    assert set(MODELS.keys()) == {1, 3, 7}


def test_prediction_probability_range():
    result = predict_unrest(sample_features(), 1)
    assert 0.0 <= result["probability"] <= 1.0


def test_all_horizons():
    for horizon in [1, 3, 7]:
        result = predict_unrest(sample_features(), horizon)
        assert result["horizon_days"] == horizon
        assert result["threshold"] == THRESHOLDS[str(horizon)]
        assert result["prediction"] in {0, 1}


def test_prediction_output():
    result = predict_unrest(sample_features(), 1)
    assert "probability" in result
    assert "prediction" in result
    assert "threshold" in result
    assert "forecast" in result
    assert result["forecast"] in {"Unrest event expected", "No unrest event expected"}


def test_missing_features():
    with pytest.raises(ValueError, match="Missing required features"):
        predict_unrest({"lag_1": 2}, 1)


def test_invalid_horizon():
    for bad_h in [0, 2, 5, 14, -1]:
        with pytest.raises(ValueError, match="Invalid horizon"):
            predict_unrest(sample_features(), horizon=bad_h)


def test_negative_feature_value():
    feats = sample_features()
    feats["lag_1"] = -3.0
    with pytest.raises(ValueError, match="cannot be negative"):
        predict_unrest(feats, 1)


def test_nan_feature_value():
    feats = sample_features()
    feats["rolling_7_mean"] = float("nan")
    with pytest.raises(ValueError, match="NaN or None"):
        predict_unrest(feats, 1)


def test_non_numeric_feature_value():
    feats = sample_features()
    feats["lag_7"] = "severe_protest"
    with pytest.raises(TypeError, match="must be numeric"):
        predict_unrest(feats, 1)


def test_non_dict_payload():
    with pytest.raises(TypeError, match="must be a dictionary"):
        predict_unrest([1, 2, 3], 1)


def test_zero_activity_forecast():
    zero_feats = {f: 0.0 for f in FEATURES}
    result = predict_unrest(zero_feats, 1)
    assert 0.0 <= result["probability"] <= 1.0
    assert result["prediction"] in {0, 1}


def test_predict_unrest_batch():
    rows = [sample_features() for _ in range(5)]
    df = pd.DataFrame(rows)

    batch_res = predict_unrest_batch(df, horizon=1)
    assert len(batch_res) == 5
    assert "probability" in batch_res.columns
    assert "prediction" in batch_res.columns
    assert "forecast" in batch_res.columns
    assert batch_res["probability"].between(0.0, 1.0).all()


def test_predict_unrest_batch_missing_column():
    df = pd.DataFrame([{"lag_1": 1.0}])
    with pytest.raises(ValueError, match="Missing required feature columns"):
        predict_unrest_batch(df, horizon=1)
