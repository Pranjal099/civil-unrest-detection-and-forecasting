
import json
import pandas as pd
from pathlib import Path

INPUT = "data/farmers-protest-tweets-2021-03-5.json"
OUTPUT = "data/farmers_protest_tweets.csv"

rows = []
total = 0

with open(INPUT, "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue

        try:
            tweet = json.loads(line)
        except json.JSONDecodeError:
            continue

        total += 1
        user = tweet.get("user") or {}

        rows.append({
            "tweet_id": tweet.get("id"),
            "text": tweet.get("content"),
            "date": tweet.get("date"),
            "username": user.get("username"),
            "url": tweet.get("url"),
            "language": tweet.get("lang"),
            "likes": tweet.get("likeCount", 0),
            "retweets": tweet.get("retweetCount", 0),
            "replies": tweet.get("replyCount", 0)
        })

        if len(rows) >= 10000:
            df = pd.DataFrame(rows)
            df.to_csv(
                OUTPUT,
                mode="a",
                index=False,
                header=not Path(OUTPUT).exists()
            )
            rows = []

if rows:
    pd.DataFrame(rows).to_csv(
        OUTPUT,
        mode="a",
        index=False,
        header=not Path(OUTPUT).exists()
    )

print("Total records processed:", total)
print("Saved:", OUTPUT)

