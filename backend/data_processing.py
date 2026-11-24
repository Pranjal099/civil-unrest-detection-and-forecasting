import logging
from pathlib import Path
from typing import List, Optional, Union

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT_DIR / "data" / "ccc_compiled_20212024.csv"

FEATURES: List[str] = [
    "lag_1",
    "lag_3",
    "lag_7",
    "lag_14",
    "rolling_7_mean",
    "rolling_7_sum",
    "rolling_14_mean",
]

TARGET_COLUMNS: List[str] = [
    "target_1day",
    "target_3day",
    "target_7day",
]


def engineer_temporal_features(
    daily_events: pd.DataFrame,
    horizons: Optional[List[int]] = None
) -> pd.DataFrame:
    """
    Generate historical lag/rolling features and future target indicators
    for state-level time-series event records.

    Parameters
    ----------
    daily_events : pd.DataFrame
        DataFrame sorted by state and date, containing ['state', 'date', 'event_count'].
    horizons : list of int, optional
        Forecast horizons in days (default: [1, 3, 7]).

    Returns
    -------
    pd.DataFrame
        Processed DataFrame with all temporal features and target columns,
        omitting edge rows with NaN values from shifting or rolling windows.
    """
    if horizons is None:
        horizons = [1, 3, 7]

    df = daily_events.copy()
    grouped = df.groupby("state")["event_count"]

    # Historical temporal features (strictly past-only, shifted by 1+)
    df["lag_1"] = grouped.shift(1)
    df["lag_3"] = grouped.shift(3)
    df["lag_7"] = grouped.shift(7)
    df["lag_14"] = grouped.shift(14)

    df["rolling_7_mean"] = grouped.transform(
        lambda x: x.shift(1).rolling(7).mean()
    )
    df["rolling_7_sum"] = grouped.transform(
        lambda x: x.shift(1).rolling(7).sum()
    )
    df["rolling_14_mean"] = grouped.transform(
        lambda x: x.shift(1).rolling(14).mean()
    )

    # Forward-looking target indicators
    grouped = df.groupby("state")["event_count"]
    target_cols = []
    for horizon in horizons:
        col_name = f"target_{horizon}day"
        target_cols.append(col_name)
        future_events = grouped.transform(
            lambda x: x.rolling(
                window=horizon,
                min_periods=horizon
            ).sum().shift(-horizon)
        )
        df[col_name] = (
            (future_events > 0)
            .where(future_events.notna())
            .astype("Int64")
        )

    # Drop edge rows with insufficient history or missing future labels
    cleaned_df = df.dropna(subset=FEATURES + target_cols).reset_index(drop=True)
    return cleaned_df


def load_and_process_data(
    data_path: Optional[Union[str, Path]] = None,
    start_date: str = "2021-01-01",
    end_date: str = "2024-12-31"
) -> pd.DataFrame:
    """
    Load raw Crowd Counting Consortium (CCC) protest data, construct
    a complete calendar multi-index across all states, and compute
    temporal forecasting features.

    Parameters
    ----------
    data_path : str or Path, optional
        Path to the CCC dataset CSV file (defaults to data/ccc_compiled_20212024.csv).
    start_date : str, default '2021-01-01'
        Calendar start date.
    end_date : str, default '2024-12-31'
        Calendar end date.

    Returns
    -------
    pd.DataFrame
        Cleaned, feature-engineered dataset ready for model training or inference.
    """
    path = Path(data_path) if data_path else DATA_PATH

    if not path.exists():
        raise FileNotFoundError(f"Dataset not found at expected location: {path}")

    logger.info("Loading CCC dataset from %s...", path)
    df = pd.read_csv(
        path,
        encoding="latin-1",
        usecols=["date", "state"]
    )

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date", "state"])

    # Aggregate event count per state per calendar day
    daily_events = (
        df.groupby(["state", "date"])
        .size()
        .reset_index(name="event_count")
    )

    # Construct complete date-state cross-product index
    dates = pd.date_range(start=start_date, end=end_date, freq="D")
    states = sorted(df["state"].unique())

    full_index = pd.MultiIndex.from_product(
        [states, dates],
        names=["state", "date"]
    )

    # Reindex missing state-date pairs to 0 events
    daily_events = (
        daily_events.set_index(["state", "date"])
        .reindex(full_index, fill_value=0)
        .reset_index()
    )

    daily_events = daily_events.sort_values(
        ["state", "date"]
    ).reset_index(drop=True)

    # Compute lags, rolling stats, and target indicators
    processed_df = engineer_temporal_features(daily_events)
    logger.info("Processed dataset shape: %s", processed_df.shape)
    return processed_df


if __name__ == "__main__":
    data = load_and_process_data()
    print(f"\nProcessed dataset: {data.shape}")
    print(f"Columns: {data.columns.tolist()}")
    print(f"Sample:\n{data.head(5).to_string()}")
    print(f"2024 rows: {len(data[data['date'].dt.year == 2024])}")
