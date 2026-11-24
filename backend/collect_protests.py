import feedparser
import pandas as pd
import urllib.parse
import time
from pathlib import Path
from datetime import datetime, timezone

EVENTS = {
    "Nepal_GenZ_2025": "Nepal Gen Z protests September 2025",
    "Bangladesh_Student_2024": "Bangladesh student uprising July August 2024",
    "India_Farmers_2020_2021": "India farmers protest 2020 2021"
}

OUTPUT = "data/protest_articles.csv"
rows = []

for event, query in EVENTS.items():
    print(f"\nCollecting: {event}")

    url = (
        "https://news.google.com/rss/search?q="
        + urllib.parse.quote(query)
        + "&hl=en-IN&gl=IN&ceid=IN:en"
    )

    feed = feedparser.parse(url)

    print("Articles found:", len(feed.entries))

    for article in feed.entries:
        source = article.get("source", {}).get("title", "Unknown")

        rows.append({
            "event": event,
            "text": article.get("title", ""),
            "source": source,
            "url": article.get("link", ""),
            "published_at": article.get("published", ""),
            "collected_at": datetime.now(timezone.utc).isoformat()
        })

    time.sleep(2)

df = pd.DataFrame(rows)

if not df.empty:
    df = df.drop_duplicates(subset=["url"])

Path("data").mkdir(exist_ok=True)
df.to_csv(OUTPUT, index=False)

print("\n========== COLLECTION SUMMARY ==========")
print("Total articles:", len(df))

if not df.empty:
    print("\nArticles per protest:")
    print(df.groupby("event").size())

print("\nSaved to:", OUTPUT)
