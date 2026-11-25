"""
social_media_features.py
Converts structured NLP extraction records into temporal region-aggregated features.

NOTE: This module is designed to work with output from extract_event() in
social_media_extraction.py, specifically for the India Farmers' Protest case study.
"""
import logging
from typing import Union

import pandas as pd

logger = logging.getLogger(__name__)

PROTEST_TYPES = {"protest", "demonstration", "march"}
STRIKE_TYPES = {"strike", "labor dispute"}
CLASH_TYPES = {"riot"}
VIOLENCE_ESCALATION_KEYWORDS = {"violence", "clash"}

SOCIAL_FEATURE_COLUMNS = [
    "sm_event_count",
    "sm_mobilization_count",
    "sm_escalation_count",
    "sm_protest_count",
    "sm_strike_count",
    "sm_clash_count",
    "sm_rolling_7_mean",
    "sm_rolling_7_sum",
    "sm_delta_vs_prev7"
]


def _parse_list_field(value) -> list:
    """
    Safely parse a field that could be a list, a string representation of a list,
    a comma-separated string, or None. Returns a list.
    """
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, str):
        v = value.strip()
        if not v:
            return []
        # Handle Python-repr lists like "['violence', 'clash']"
        if v.startswith("[") and v.endswith("]"):
            try:
                import ast
                parsed = ast.literal_eval(v)
                return list(parsed) if isinstance(parsed, (list, tuple)) else [str(parsed)]
            except (ValueError, SyntaxError):
                pass
        # Fallback: split on comma
        return [item.strip().strip("'\"") for item in v.split(",") if item.strip()]
    return []


def compute_social_features(
    extractions_df: pd.DataFrame,
    date_col: str = "date",
) -> pd.DataFrame:
    """
    Given a DataFrame of NLP extraction results, compute daily aggregated features.

    Expected columns (at minimum):
      - date (datetime or parseable string)
      - event_detected (bool or 0/1)
      - event_type (str or None)
      - escalation_indicators (list or string representation of list)
      - mobilization_call (bool or 0/1)
      - locations (list or string representation of list)

    Returns
    -------
    pd.DataFrame
        Indexed by date (daily), with columns:
          sm_event_count, sm_mobilization_count, sm_escalation_count,
          sm_protest_count, sm_strike_count, sm_clash_count,
          sm_rolling_7_mean, sm_rolling_7_sum, sm_delta_vs_prev7
    """
    if extractions_df.empty:
        logger.warning("compute_social_features called with empty DataFrame.")
        return pd.DataFrame(columns=[
            "sm_event_count", "sm_mobilization_count", "sm_escalation_count",
            "sm_protest_count", "sm_strike_count", "sm_clash_count",
            "sm_rolling_7_mean", "sm_rolling_7_sum", "sm_delta_vs_prev7",
        ])

    df = extractions_df.copy()

    # Normalize date column
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce", utc=True)
    df["_date_only"] = df[date_col].dt.normalize()

    # Normalize boolean fields
    df["_event_detected"] = df["event_detected"].fillna(False).astype(bool)
    df["_mobilization"] = df["mobilization_call"].fillna(False).astype(bool) if "mobilization_call" in df.columns else False

    # Parse escalation_indicators
    if "escalation_indicators" in df.columns:
        df["_has_escalation"] = df["escalation_indicators"].apply(
            lambda x: len(_parse_list_field(x)) > 0
        )
        df["_has_violence_clash"] = df["escalation_indicators"].apply(
            lambda x: bool(VIOLENCE_ESCALATION_KEYWORDS.intersection(
                {kw.lower() for kw in _parse_list_field(x)}
            ))
        )
    else:
        df["_has_escalation"] = False
        df["_has_violence_clash"] = False

    # Normalize event_type to lowercase
    if "event_type" in df.columns:
        df["_event_type"] = df["event_type"].apply(
            lambda x: x.lower().strip() if isinstance(x, str) else None
        )
    else:
        df["_event_type"] = None

    df["_is_protest"] = df["_event_type"].apply(lambda x: x in PROTEST_TYPES if x else False)
    df["_is_strike"] = df["_event_type"].apply(lambda x: x in STRIKE_TYPES if x else False)
    df["_is_clash"] = df.apply(
        lambda row: (row["_event_type"] in CLASH_TYPES if row["_event_type"] else False)
        or row["_has_violence_clash"],
        axis=1,
    )

    # Daily aggregation
    daily = df.groupby("_date_only").agg(
        sm_event_count=("_event_detected", "sum"),
        sm_mobilization_count=("_mobilization", "sum"),
        sm_escalation_count=("_has_escalation", "sum"),
        sm_protest_count=("_is_protest", "sum"),
        sm_strike_count=("_is_strike", "sum"),
        sm_clash_count=("_is_clash", "sum"),
    ).astype(int)

    daily.index.name = "date"
    daily = daily.sort_index()

    # Rolling features (min_periods=1 to avoid NaN on early days)
    daily["sm_rolling_7_mean"] = (
        daily["sm_event_count"].rolling(window=7, min_periods=1).mean().round(4)
    )
    daily["sm_rolling_7_sum"] = (
        daily["sm_event_count"].rolling(window=7, min_periods=1).sum().astype(int)
    )

    # Delta vs previous 7-day mean: difference from the mean of the 7 days before this day
    # For day T: mean of days [T-7, T-1] minus mean of days [T-14, T-8]
    prev7_mean = (
        daily["sm_event_count"]
        .shift(1)
        .rolling(window=7, min_periods=1)
        .mean()
    )
    daily["sm_delta_vs_prev7"] = (daily["sm_rolling_7_mean"] - prev7_mean).round(4)
    daily["sm_delta_vs_prev7"] = daily["sm_delta_vs_prev7"].fillna(0.0)

    return daily


def compute_daily_intensity_target(df: pd.DataFrame, threshold: int = 10) -> pd.Series:
    """
    Binary target: was there a spike in social media activity the next day?

    Given a daily feature DataFrame (output of compute_social_features),
    returns a binary Series where 1 = next-day sm_event_count is above the
    median of the whole series. This is a purely within-dataset, within-protest
    prediction task for the India Farmers' Protest case study ONLY.

    Parameters
    ----------
    df : pd.DataFrame
        Output of compute_social_features (indexed by date).
    threshold : int
        Unused — kept for API compatibility. The actual threshold is the
        dataset median of sm_event_count.

    Returns
    -------
    pd.Series
        Binary Series (0/1), indexed by date, representing next-day activity spike.
        Last day is NaN (no next-day label available).

    Notes
    -----
    This target is used ONLY for the India case study. It does NOT relate to
    CCC-based civil unrest forecasting.
    """
    if df.empty or "sm_event_count" not in df.columns:
        return pd.Series(dtype=int)

    next_day_count = df["sm_event_count"].shift(-1)
    median_count = df["sm_event_count"].median()
    target = (next_day_count > median_count).astype("Int64")
    target.name = "next_day_spike"
    return target


def create_social_feature_vector(
    daily_features_df: pd.DataFrame,
    as_of_date = None
) -> dict:
    """
    Extract a point-in-time social media feature vector for a given date.
    """
    if daily_features_df.empty:
        return {col: 0.0 for col in SOCIAL_FEATURE_COLUMNS}

    df = daily_features_df.copy()
    if as_of_date is not None:
        target_ts = pd.to_datetime(as_of_date, utc=True).normalize()
        if target_ts in df.index:
            row = df.loc[target_ts]
        else:
            row = df.iloc[-1]
    else:
        row = df.iloc[-1]

    return {
        col: float(row.get(col, 0.0))
        for col in SOCIAL_FEATURE_COLUMNS
    }
