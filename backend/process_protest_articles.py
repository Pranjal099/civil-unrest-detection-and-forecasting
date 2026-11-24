import pandas as pd
from pathlib import Path
from backend.social_media_extraction import extract_event

INPUT = "data/final_protest_articles.csv"
OUTPUT = "results/final_protest_extractions.csv"

df = pd.read_csv(INPUT)
results = []

for i, row in df.iterrows():
    text = str(row["text"])
    extracted = extract_event(text)

    results.append({
        "event": row["event"],
        "source": row["source"],
        "url": row["url"],
        "published_at": row["published_at"],
        **extracted
    })

output = pd.DataFrame(results)
Path("results").mkdir(exist_ok=True)
output.to_csv(OUTPUT, index=False)

print("\n========== NLP EXTRACTION ==========")
print("Articles processed:", len(output))
print("Events detected:", int(output["event_detected"].sum()))
print("Non-events:", int((~output["event_detected"]).sum()))

print("\nDetection by protest:")
print(output.groupby("event")["event_detected"].agg(["count", "sum"]))

print("\nEvent types:")
print(output.loc[output["event_detected"], "event_type"].value_counts())

print("\nSaved:", OUTPUT)
