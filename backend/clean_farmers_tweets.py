
import pandas as pd

INPUT = "data/farmers_protest_tweets.csv"
OUTPUT = "data/cleaned_farmers_protest_tweets.csv"

df = pd.read_csv(INPUT)

before = len(df)

df = df.dropna(subset=["text"])
df["text"] = df["text"].astype(str).str.strip()
df = df[df["text"] != ""]
df = df.drop_duplicates(subset=["tweet_id"])
df = df[df["language"] == "en"]

df.to_csv(OUTPUT, index=False)

print("Original tweets:", before)
print("Clean tweets:", len(df))
print("Duplicates/invalid/non-English removed:", before - len(df))
print("Saved:", OUTPUT)
