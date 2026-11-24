import pytest
import pandas as pd
import numpy as np

from backend.data_processing import (
    FEATURES,
    TARGET_COLUMNS,
    engineer_temporal_features
)


@pytest.fixture
def synthetic_daily_events():
    """
    Construct deterministic daily event counts for two states
    to test lag, rolling window, and target math without external files.
    """
    dates = pd.date_range("2023-01-01", "2023-02-28", freq="D")
    records = []
    for state, base_val in [("NY", 2), ("TX", 0)]:
        for i, dt in enumerate(dates):
            # Deterministic count: NY has events on even days, TX on days divisible by 5
            count = base_val if i % 2 == 0 else 0
            records.append({
                "state": state,
                "date": dt,
                "event_count": count
            })
    df = pd.DataFrame(records).sort_values(["state", "date"]).reset_index(drop=True)
    return df


def test_engineer_temporal_features_structure(synthetic_daily_events):
    processed = engineer_temporal_features(synthetic_daily_events)

    for feat in FEATURES:
        assert feat in processed.columns, f"Missing feature: {feat}"
    for target in TARGET_COLUMNS:
        assert target in processed.columns, f"Missing target: {target}"

    assert len(processed) > 0
    assert not processed[FEATURES].isna().any().any()
    assert not processed[TARGET_COLUMNS].isna().any().any()


def test_lag_mathematical_correctness():
    """Verify that lag_k strictly refers to the event count k days prior."""
    dates = pd.date_range("2023-01-01", "2023-01-31", freq="D")
    df = pd.DataFrame({
        "state": ["CA"] * len(dates),
        "date": dates,
        "event_count": list(range(len(dates)))  # 0, 1, 2, ...
    })

    processed = engineer_temporal_features(df)
    for _, row in processed.iterrows():
        orig_idx = df[df["date"] == row["date"]].index[0]
        # lag_1 should be event_count at orig_idx - 1
        assert row["lag_1"] == df.loc[orig_idx - 1, "event_count"]
        # lag_7 should be event_count at orig_idx - 7
        assert row["lag_7"] == df.loc[orig_idx - 7, "event_count"]


def test_no_lookahead_leakage():
    """
    Verify that current day's event count does NOT leak into
    lag_1 or rolling_7_mean for the current day.
    """
    dates = pd.date_range("2023-01-01", "2023-01-31", freq="D")
    df1 = pd.DataFrame({
        "state": ["CA"] * len(dates),
        "date": dates,
        "event_count": [1] * len(dates)
    })
    # df2 has identical history except today (index 20) has an enormous surge
    df2 = df1.copy()
    df2.loc[20, "event_count"] = 500

    p1 = engineer_temporal_features(df1)
    p2 = engineer_temporal_features(df2)

    # At index 20, past features (lag_1, rolling_7_mean) MUST be identical
    row1 = p1[p1["date"] == dates[20]].iloc[0]
    row2 = p2[p2["date"] == dates[20]].iloc[0]

    assert row1["lag_1"] == row2["lag_1"]
    assert row1["rolling_7_mean"] == row2["rolling_7_mean"]
    assert row1["rolling_14_mean"] == row2["rolling_14_mean"]


def test_target_generation_logic():
    """Verify that forward-looking targets indicate event occurrences strictly in the future."""
    dates = pd.date_range("2023-01-01", "2023-01-25", freq="D")
    # All zeroes except an event on day 18
    counts = [0] * len(dates)
    counts[18] = 3

    df = pd.DataFrame({
        "state": ["WA"] * len(dates),
        "date": dates,
        "event_count": counts
    })
    processed = engineer_temporal_features(df)

    # On day 17, target_1day should be 1 because day 18 has an event
    day17 = processed[processed["date"] == dates[17]]
    if not day17.empty:
        assert day17.iloc[0]["target_1day"] == 1

    # On day 15, target_1day should be 0 because day 16 has 0 events
    day15 = processed[processed["date"] == dates[15]]
    if not day15.empty:
        assert day15.iloc[0]["target_1day"] == 0
        # But target_3day on day 15 covers day 16, 17, 18, so it must be 1
        assert day15.iloc[0]["target_3day"] == 1


def test_state_isolation():
    """Ensure lags and rolling features for State A never spill into State B."""
    dates = pd.date_range("2023-01-01", "2023-01-25", freq="D")
    # State A has high activity, State B has 0 activity
    records = []
    for dt in dates:
        records.append({"state": "A", "date": dt, "event_count": 10})
        records.append({"state": "B", "date": dt, "event_count": 0})

    df = pd.DataFrame(records).sort_values(["state", "date"]).reset_index(drop=True)
    processed = engineer_temporal_features(df)

    b_rows = processed[processed["state"] == "B"]
    assert (b_rows["lag_1"] == 0).all()
    assert (b_rows["rolling_7_mean"] == 0).all()
    assert (b_rows["target_1day"] == 0).all()
