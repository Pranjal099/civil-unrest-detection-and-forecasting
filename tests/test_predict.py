from backend.predict import predict_unrest, FEATURES, MODELS


def sample_features():
    return {
        "lag_1": 2,
        "lag_3": 5,
        "lag_7": 3,
        "lag_14": 4,
        "rolling_7_mean": 1.5,
        "rolling_7_sum": 10,
        "rolling_14_mean": 1.2
    }


def test_models_loaded():
    assert set(MODELS.keys()) == {1, 3, 7}


def test_prediction_probability_range():
    result = predict_unrest(sample_features(), 1)
    assert 0 <= result["probability"] <= 1


def test_all_horizons():
    for horizon in [1, 3, 7]:
        result = predict_unrest(sample_features(), horizon)
        assert result["horizon_days"] == horizon


def test_prediction_output():
    result = predict_unrest(sample_features(), 1)
    assert "probability" in result
    assert "prediction" in result
    assert result["prediction"] in [0, 1]


def test_missing_features():
    try:
        predict_unrest({"lag_1": 2}, 1)
    except ValueError:
        pass
    else:
        assert False, "Expected ValueError for missing features"