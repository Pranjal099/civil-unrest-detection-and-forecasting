import pytest
import pandas as pd
import numpy as np
from pathlib import Path

from backend.train import (
    prepare_temporal_splits,
    train_single_horizon,
    evaluate_horizon_model,
    run_training_pipeline
)
from backend.data_processing import FEATURES


@pytest.fixture
def synthetic_training_data():
    """Generate minimal synthetic time-series data covering 2021-2024."""
    dates = pd.date_range("2021-01-01", "2024-12-31", freq="W")
    n = len(dates)
    np.random.seed(42)

    data = {
        "date": dates,
        "state": ["CA"] * n,
    }
    for feat in FEATURES:
        data[feat] = np.random.poisson(lam=2, size=n).astype(float)

    for h in [1, 3, 7]:
        data[f"target_{h}day"] = (np.random.rand(n) > 0.6).astype(int)

    return pd.DataFrame(data)


def test_prepare_temporal_splits(synthetic_training_data):
    splits = prepare_temporal_splits(synthetic_training_data, FEATURES, "target_1day")

    assert set(splits.keys()) == {"train", "val", "test"}

    X_train, y_train = splits["train"]
    X_val, y_val = splits["val"]
    X_test, y_test = splits["test"]

    assert len(X_train) == len(y_train)
    assert len(X_val) == len(y_val)
    assert len(X_test) == len(y_test)
    assert len(X_train) > 0 and len(X_val) > 0 and len(X_test) > 0
    assert list(X_train.columns) == FEATURES


def test_train_single_horizon(synthetic_training_data):
    splits = prepare_temporal_splits(synthetic_training_data, FEATURES, "target_1day")
    X_train, y_train = splits["train"]

    model = train_single_horizon(X_train, y_train, n_estimators=10, max_depth=3)
    assert hasattr(model, "predict_proba")

    probs = model.predict_proba(X_train)[:, 1]
    assert len(probs) == len(X_train)
    assert np.all((probs >= 0.0) & (probs <= 1.0))


def test_evaluate_horizon_model(synthetic_training_data):
    splits = prepare_temporal_splits(synthetic_training_data, FEATURES, "target_1day")
    X_train, y_train = splits["train"]
    X_test, y_test = splits["test"]

    model = train_single_horizon(X_train, y_train, n_estimators=10, max_depth=3)
    metrics = evaluate_horizon_model(model, X_test, y_test, threshold=0.35)

    assert "pr_auc" in metrics
    assert "f1" in metrics
    assert "precision" in metrics
    assert "recall" in metrics
    assert "accuracy" in metrics
    for metric_name, val in metrics.items():
        assert 0.0 <= val <= 1.0, f"{metric_name} out of bounds: {val}"


def test_run_training_pipeline_dry_run(synthetic_training_data):
    results = run_training_pipeline(data=synthetic_training_data, dry_run=True, horizons=[1])
    assert 1 in results
    assert "train_samples" in results[1]
    assert "test_samples" in results[1]


def test_run_training_pipeline_save_to_tmp(synthetic_training_data, tmp_path):
    # Ensure temporary directory is used and existing models directory is never touched
    results = run_training_pipeline(
        data=synthetic_training_data,
        output_dir=tmp_path,
        dry_run=False,
        horizons=[1]
    )
    assert 1 in results
    assert (tmp_path / "xgboost_1day.json").exists()
