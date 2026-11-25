"""
test_integration.py
-------------------
Unit and integration tests for:
- backend.social_media_features
- backend.integration_pipeline
- backend.case_study
"""

import pytest
import pandas as pd
import numpy as np

from backend.social_media_features import (
    compute_social_features,
    create_social_feature_vector,
    SOCIAL_FEATURE_COLUMNS,
)
from backend.integration_pipeline import IntegrationPipeline
from backend.case_study import run_case_study


def test_social_features_empty_dataframe():
    empty_df = pd.DataFrame()
    res = compute_social_features(empty_df)
    assert isinstance(res, pd.DataFrame)
    for col in SOCIAL_FEATURE_COLUMNS:
        assert col in res.columns or res.empty


def test_social_features_computation_logic():
    data = [
        {
            "date": "2021-02-01",
            "event_detected": True,
            "event_type": "protest",
            "escalation_indicators": ["clash"],
            "mobilization_call": True,
            "actors": ["farmers", "BKU"],
        },
        {
            "date": "2021-02-01",
            "event_detected": True,
            "event_type": "strike",
            "escalation_indicators": [],
            "mobilization_call": False,
            "actors": ["workers"],
        },
        {
            "date": "2021-02-01",
            "event_detected": False,
            "event_type": None,
            "escalation_indicators": [],
            "mobilization_call": False,
            "actors": [],
        },
        {
            "date": "2021-02-02",
            "event_detected": True,
            "event_type": "protest",
            "escalation_indicators": ["teargas"],
            "mobilization_call": True,
            "actors": ["students"],
        },
    ]
    df = pd.DataFrame(data)
    features = compute_social_features(df)

    assert not features.empty
    assert len(features) == 2  # Two distinct days
    assert "sm_event_count" in features.columns
    assert "sm_mobilization_count" in features.columns
    assert "sm_escalation_count" in features.columns
    assert "sm_rolling_7_mean" in features.columns

    # Day 1 checks: 2 detected events, 1 mobilization, 1 escalation, 1 protest, 1 strike
    day1 = features.iloc[0]
    assert day1["sm_event_count"] == 2
    assert day1["sm_mobilization_count"] == 1
    assert day1["sm_escalation_count"] == 1
    assert day1["sm_protest_count"] == 1
    assert day1["sm_strike_count"] == 1


def test_create_social_feature_vector():
    data = [
        {
            "date": "2021-02-01",
            "event_detected": True,
            "event_type": "protest",
            "escalation_indicators": ["clash"],
            "mobilization_call": True,
            "actors": ["farmers"],
        }
    ]
    df = pd.DataFrame(data)
    features = compute_social_features(df)
    vec = create_social_feature_vector(features, as_of_date="2021-02-01")

    assert isinstance(vec, dict)
    for col in SOCIAL_FEATURE_COLUMNS:
        assert col in vec
        assert isinstance(vec[col], (int, float))


def test_integration_pipeline_run_demo_mock():
    pipeline = IntegrationPipeline(threshold=0.70)
    texts = [
        "Farmers announced a march to Delhi border tomorrow.",
        "Beautiful sunny day for a walk in the park."
    ]
    dates = ["2021-02-10", "2021-02-10"]

    # Test run_demo output structure without calling heavy models if possible
    # IntegrationPipeline uses extract_event which works
    res = pipeline.run_demo(texts, dates)
    assert "n_texts" in res
    assert res["n_texts"] == 2
    assert "n_events_detected" in res
    assert "daily_features" in res
    assert "disclaimer" in res


def test_run_case_study_synthetic():
    # Build 30 days of synthetic extraction events
    rng = np.random.RandomState(42)
    rows = []
    base_date = pd.Timestamp("2021-02-01")
    for d in range(30):
        day_date = str((base_date + pd.Timedelta(days=d)).date())
        n_posts = rng.randint(5, 20)
        for _ in range(n_posts):
            detected = bool(rng.rand() > 0.4)
            rows.append({
                "date": day_date,
                "event_detected": detected,
                "event_type": rng.choice(["protest", "strike", "riot", None]),
                "escalation_indicators": ["clash"] if rng.rand() > 0.7 else [],
                "mobilization_call": bool(rng.rand() > 0.6),
                "actors": ["farmers"] if detected else []
            })
    df_synth = pd.DataFrame(rows)
    results = run_case_study(df_synth, train_ratio=0.7, save_results=False)

    assert "model_A_baseline" in results
    assert "model_B_integrated" in results
    assert "comparison_delta_B_minus_A" in results
    assert "metrics" in results["model_A_baseline"]
    assert "metrics" in results["model_B_integrated"]
    for m in ["accuracy", "precision", "recall", "f1", "pr_auc"]:
        assert m in results["model_A_baseline"]["metrics"]
        assert m in results["model_B_integrated"]["metrics"]
