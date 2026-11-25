"""
integration_pipeline.py
Demo integration pipeline: raw social media text → NLP extraction → feature vectors.

This module is the demo integration path for the dashboard. It runs extract_event()
on raw text, computes social media features via compute_social_features(), and
returns structured feature vectors.

IMPORTANT: This pipeline does NOT claim to improve CCC XGBoost model performance.
It is a standalone demonstration of how social media signals can be structured for
civil unrest monitoring. The CCC forecasting models are independent.
"""
import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Union

import pandas as pd

from backend.social_media_extraction import extract_event
from backend.social_media_features import compute_social_features

logger = logging.getLogger(__name__)


class IntegrationPipeline:
    """
    End-to-end pipeline: raw social media posts → structured NLP extractions → feature vectors.

    Usage
    -----
    >>> pipeline = IntegrationPipeline()
    >>> result = pipeline.run_demo(texts=["Farmers protest in Delhi"], dates=["2021-02-10"])
    >>> print(result)

    Notes
    -----
    - extract_event() uses lazy-loaded models (BART-large-MNLI + spaCy).
    - Each text is processed independently (no batch optimization).
    - For large-scale processing (>10K texts), use GPU or batch APIs instead.
    - This pipeline does NOT feed into the CCC XGBoost forecasting pipeline.
    """

    def __init__(self, threshold: float = 0.70):
        """
        Parameters
        ----------
        threshold : float
            Confidence threshold for binary event detection (passed to extract_event).
        """
        self.threshold = threshold

    def run_extraction(
        self,
        texts: List[str],
        dates: List[Union[str, datetime]],
        sources: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """
        Extract civil unrest events from a list of raw text strings.

        Parameters
        ----------
        texts : list of str
            Raw social media text strings.
        dates : list of str or datetime
            Corresponding dates/timestamps for each text.
        sources : list of str, optional
            Optional source labels (e.g., "twitter", "news"). If None, all set to None.

        Returns
        -------
        pd.DataFrame
            One row per input text with extraction results plus 'date' and 'source' columns.
        """
        if len(texts) != len(dates):
            raise ValueError(
                f"texts and dates must have equal length: {len(texts)} vs {len(dates)}"
            )
        if sources is None:
            sources = [None] * len(texts)
        elif len(sources) != len(texts):
            raise ValueError(
                f"sources must have same length as texts: {len(sources)} vs {len(texts)}"
            )

        rows = []
        total = len(texts)
        for i, (text, date, source) in enumerate(zip(texts, dates, sources)):
            if i % 100 == 0 and i > 0:
                logger.info("Extracting: %d/%d", i, total)
            try:
                result = extract_event(text, threshold=self.threshold)
            except Exception as exc:
                logger.warning("extract_event failed for text[%d]: %s", i, exc)
                result = {
                    "text": str(text) if text is not None else "",
                    "event_detected": False,
                    "confidence": 0.0,
                    "event_type": None,
                    "locations": [],
                    "organizations": [],
                    "dates": [],
                }

            result["date"] = date
            result["source"] = source
            # Ensure required fields exist for compute_social_features
            result.setdefault("escalation_indicators", [])
            result.setdefault("mobilization_call", False)
            rows.append(result)

        df = pd.DataFrame(rows)
        df["date"] = pd.to_datetime(df["date"], errors="coerce", utc=True)
        return df

    def get_features_for_date(
        self,
        extractions_df: pd.DataFrame,
        as_of_date: Union[str, datetime, pd.Timestamp],
    ) -> Dict[str, Any]:
        """
        Get the social media feature vector for a specific date.

        Parameters
        ----------
        extractions_df : pd.DataFrame
            Output of run_extraction().
        as_of_date : str, datetime, or Timestamp
            The date to retrieve features for.

        Returns
        -------
        dict
            Feature dictionary for the given date, or zeros if date not in data.
        """
        features_df = compute_social_features(extractions_df)

        as_of_ts = pd.to_datetime(as_of_date, utc=True).normalize()

        if as_of_ts in features_df.index:
            row = features_df.loc[as_of_ts]
            return row.to_dict()
        else:
            # Return zero vector for missing date
            logger.warning("Date %s not found in features; returning zero vector.", as_of_ts)
            zero = {col: 0 for col in features_df.columns}
            return zero

    def run_demo(
        self,
        texts: List[str],
        dates: List[Union[str, datetime]],
        sources: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Demo mode: run full pipeline and return a structured summary.

        Steps:
        1. Extract events from texts
        2. Compute daily social media features
        3. Return summary dict

        Parameters
        ----------
        texts : list of str
            Raw social media texts.
        dates : list of str or datetime
            Corresponding dates.
        sources : list of str, optional
            Optional source labels.

        Returns
        -------
        dict with keys:
            - n_texts: number of input texts
            - n_events_detected: number with event_detected=True
            - detection_rate: fraction detected
            - date_range: [min_date, max_date]
            - daily_features: dict of {date_str: feature_dict}
            - feature_columns: list of feature column names
            - disclaimer: methodological note
        """
        extractions_df = self.run_extraction(texts, dates, sources)
        features_df = compute_social_features(extractions_df)

        n_texts = len(texts)
        n_events = int(extractions_df["event_detected"].sum())
        detection_rate = round(n_events / n_texts, 4) if n_texts > 0 else 0.0

        # Date range
        dates_parsed = extractions_df["date"].dropna()
        date_min = str(dates_parsed.min().date()) if not dates_parsed.empty else None
        date_max = str(dates_parsed.max().date()) if not dates_parsed.empty else None

        # Daily features as plain dict
        daily_dict: Dict[str, Any] = {}
        for idx, row in features_df.iterrows():
            date_str = str(idx.date()) if hasattr(idx, "date") else str(idx)
            daily_dict[date_str] = {k: (int(v) if isinstance(v, (int,)) else round(float(v), 4))
                                    for k, v in row.items()}

        return {
            "n_texts": n_texts,
            "n_events_detected": n_events,
            "detection_rate": detection_rate,
            "date_range": [date_min, date_max],
            "daily_features": daily_dict,
            "feature_columns": list(features_df.columns),
            "disclaimer": (
                "DEMO PIPELINE: Social media features extracted for monitoring purposes only. "
                "These features are NOT fed into the CCC XGBoost civil unrest forecasting models. "
                "The CCC models use historical protest data (US states, 2021-2024). "
                "Social media features are presented as a separate exploratory layer."
            ),
        }
