import pandas as pd
from pathlib import Path
from backend.social_media_extraction import extract_event

INPUT_FILE = "data/demo_social_posts.csv"
OUTPUT_FILE = "results/demo_extracted_events.csv"

df = pd.read_csv(INPUT_FILE)
results = []

for _, row in df.iterrows():
    extracted = extract_event(row["text"])
    results.append({
        "id": row["id"],
        **extracted
    })

output = pd.DataFrame(results)
Path("results").mkdir(exist_ok=True)
output.to_csv(OUTPUT_FILE, index=False)

print("Extraction completed!")
print("Total posts:", len(output))
print("Events detected:", int(output["event_detected"].sum()))
print("Non-events:", int((~output["event_detected"]).sum()))
print("Saved to:", OUTPUT_FILE)
print("\nResults:")
print(output[["id", "text", "event_detected", "event_type", "locations"]].to_string(index=False))
