import pandas as pd

INPUT = "data/cleaned_protest_articles.csv"
OUTPUT = "data/final_protest_articles.csv"

df = pd.read_csv(INPUT)
df["published_at"] = pd.to_datetime(df["published_at"], utc=True, errors="coerce")

windows = {
    "Nepal_GenZ_2025": ("2025-09-01", "2025-10-31"),
    "Bangladesh_Student_2024": ("2024-07-01", "2024-09-30"),
    "India_Farmers_2020_2021": ("2020-11-01", "2022-01-31"),
}

filtered = []

for event, (start, end) in windows.items():
    subset = df[
        (df["event"] == event) &
        (df["published_at"] >= pd.Timestamp(start, tz="UTC")) &
        (df["published_at"] < pd.Timestamp(end, tz="UTC") + pd.Timedelta(days=1))
    ].copy()

    filtered.append(subset)
    print(f"{event}: {len(subset)} articles")

final_df = pd.concat(filtered, ignore_index=True)
final_df.to_csv(OUTPUT, index=False)

print("\nTotal articles within event windows:", len(final_df))
print("Saved:", OUTPUT)
