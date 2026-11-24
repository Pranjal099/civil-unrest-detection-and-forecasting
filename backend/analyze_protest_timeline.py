import pandas as pd

df = pd.read_csv("data/cleaned_protest_articles.csv")

df["published_at"] = pd.to_datetime(
    df["published_at"], errors="coerce", utc=True
)

print("Missing publication dates:", df["published_at"].isna().sum())

df = df.dropna(subset=["published_at"])
df["date"] = df["published_at"].dt.date

print("\n========== PROTEST TIMELINES ==========")

for event, group in df.groupby("event"):
    print(f"\n{event}")
    timeline = group.groupby("date").size().sort_index()
    print(timeline.to_string())
    print("Total dated articles:", len(group))
    print("Date range:", group["published_at"].min(), "to", group["published_at"].max())

print("\nSaved timeline data:")
timeline_data = df.groupby(["event", "date"]).size().reset_index(name="article_count")
timeline_data.to_csv("results/protest_timeline.csv", index=False)
print("results/protest_timeline.csv")
