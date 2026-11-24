import py_compile
from pathlib import Path
import pytest
import pandas as pd

from backend.predict import FEATURES, predict_unrest, THRESHOLDS

ROOT_DIR = Path(__file__).resolve().parent.parent
APP_PATH = ROOT_DIR / "app.py"


def test_app_syntax_and_compilation():
    """Verify that app.py compiles cleanly without syntax errors."""
    assert APP_PATH.exists()
    compiled_path = py_compile.compile(str(APP_PATH), doraise=True)
    assert compiled_path is not None


def test_dashboard_features_alignment():
    """Verify that dashboard display features match backend model expectations."""
    feature_display_keys = [
        "lag_1",
        "lag_3",
        "lag_7",
        "lag_14",
        "rolling_7_mean",
        "rolling_7_sum",
        "rolling_14_mean"
    ]
    assert set(feature_display_keys) == set(FEATURES)


def test_dashboard_prediction_execution_flow():
    """Simulate user selecting a row and generating forecast results."""
    sample_row = {
        "lag_1": 1.0,
        "lag_3": 2.0,
        "lag_7": 4.0,
        "lag_14": 5.0,
        "rolling_7_mean": 2.2,
        "rolling_7_sum": 15.0,
        "rolling_14_mean": 1.9,
        "target_1day": 1,
        "target_3day": 1,
        "target_7day": 1
    }

    features = {f: float(sample_row[f]) for f in FEATURES}
    results = {h: predict_unrest(features, h) for h in [1, 3, 7]}

    assert len(results) == 3
    for h in [1, 3, 7]:
        assert results[h]["horizon_days"] == h
        assert 0.0 <= results[h]["probability"] <= 1.0
        assert results[h]["prediction"] in {0, 1}
        assert results[h]["threshold"] == THRESHOLDS[str(h)]

    # Build comparison records as rendered in app
    comparison = []
    for h in [1, 3, 7]:
        pred = results[h]["prediction"]
        act = int(sample_row[f"target_{h}day"])
        comparison.append({
            "Horizon": f"{h} Day(s)",
            "Probability": f"{results[h]['probability'] * 100:.1f}%",
            "Predicted": "Event Expected" if pred == 1 else "No Event",
            "Actual": "Event Occurred" if act == 1 else "No Event",
            "Verification": "Match" if pred == act else "Mismatch"
        })

    df_comp = pd.DataFrame(comparison)
    assert len(df_comp) == 3
    assert "Verification" in df_comp.columns
