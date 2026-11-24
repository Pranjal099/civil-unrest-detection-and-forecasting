import pandas as pd
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT_DIR / "data" / "ccc_compiled_20212024.csv"

FEATURES = [
    "lag_1",
    "lag_3",
    "lag_7",
    "lag_14",
    "rolling_7_mean",
    "rolling_7_sum",
    "rolling_14_mean",
]


def load_and_process_data():

    # Load dataset
    df = pd.read_csv(
        DATA_PATH,
        encoding="latin-1",
        usecols=["date", "state"]
    )

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date", "state"])

    # Count events per state per day
    daily_events = (
        df.groupby(["state", "date"])
        .size()
        .reset_index(name="event_count")
    )

    # Create complete calendar
    dates = pd.date_range(
        start="2021-01-01",
        end="2024-12-31",
        freq="D"
    )

    states = sorted(df["state"].unique())

    full_index = pd.MultiIndex.from_product(
        [states, dates],
        names=["state", "date"]
    )

    # Fill missing state-date combinations with zero
    daily_events = (
        daily_events.set_index(["state", "date"])
        .reindex(full_index, fill_value=0)
        .reset_index()
    )

    daily_events = daily_events.sort_values(
        ["state", "date"]
    ).reset_index(drop=True)

    # Generate historical features
    grouped = daily_events.groupby("state")["event_count"]

    daily_events["lag_1"] = grouped.shift(1)
    daily_events["lag_3"] = grouped.shift(3)
    daily_events["lag_7"] = grouped.shift(7)
    daily_events["lag_14"] = grouped.shift(14)

    daily_events["rolling_7_mean"] = grouped.transform(
        lambda x: x.shift(1).rolling(7).mean()
    )

    daily_events["rolling_7_sum"] = grouped.transform(
        lambda x: x.shift(1).rolling(7).sum()
    )

    daily_events["rolling_14_mean"] = grouped.transform(
        lambda x: x.shift(1).rolling(14).mean()
    )

    # Generate future event targets
    grouped = daily_events.groupby("state")["event_count"]

    for horizon in [1, 3, 7]:

        future_events = grouped.transform(
            lambda x: x.rolling(
                window=horizon,
                min_periods=horizon
            ).sum().shift(-horizon)
        )

        daily_events[f"target_{horizon}day"] = (
            (future_events > 0)
            .where(future_events.notna())
            .astype("Int64")
        )

    # Remove rows with insufficient history or unavailable future labels
    target_columns = [
        "target_1day",
        "target_3day",
        "target_7day"
    ]

    daily_events = daily_events.dropna(
        subset=FEATURES + target_columns
    )

    return daily_events


if __name__ == "__main__":

    data = load_and_process_data()

    print("\nProcessed dataset:", data.shape)

    print("\nColumns:")
    print(data.columns.tolist())

    print("\nSample:")
    print(data.head(5).to_string())

    print("\n2024 rows:")
    print(len(data[data["date"].dt.year == 2024]))

    print("\nTarget distribution:")
    for target in ["target_1day", "target_3day", "target_7day"]:
        print(f"\n{target}:")
        print(data[target].value_counts())