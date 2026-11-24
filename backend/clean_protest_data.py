import pandas as pd

INPUT = "data/protest_articles.csv"
OUTPUT = "data/cleaned_protest_articles.csv"

df = pd.read_csv(INPUT)

keywords = {
    "Nepal_GenZ_2025": ["nepal", "gen z", "gen-z", "genz", "youth uprising"],
    "Bangladesh_Student_2024": ["bangladesh", "student protest", "quota protest", "student uprising"],
    "India_Farmers_2020_2021": ["farmer", "farmers protest", "agriculture protest", "delhi protest"]
}

def relevant(row):
    text = str(row["text"]).lower()
    return any(word in text for word in keywords.get(row["event"], []))

df["relevant"] = df.apply(relevant, axis=1)
cleaned = df[df["relevant"]].copy()
cleaned = cleaned.drop_duplicates(subset=["url"])

cleaned.to_csv(OUTPUT, index=False)

print("Original articles:", len(df))
print("Relevant articles:", len(cleaned))
print("Removed articles:", len(df) - len(cleaned))
print("\nArticles per event:")
print(cleaned.groupby("event").size())
print("\nSaved:", OUTPUT)
