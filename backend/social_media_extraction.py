import spacy
from transformers import pipeline

nlp = spacy.load("en_core_web_sm")

# Stage 1: Civil unrest detection
detector = pipeline(
    "zero-shot-classification",
    model="facebook/bart-large-mnli"
)

# Stage 2: Event type classification
event_classifier = pipeline(
    "zero-shot-classification",
    model="facebook/bart-large-mnli"
)

EVENT_LABELS = [
    "protest",
    "demonstration",
    "strike",
    "riot",
    "labor dispute",
    "political unrest",
    "other"
]

def extract_event(text):

    # Stage 1: Binary classification
    detection = detector(
        text,
        candidate_labels=[
            "text reporting civil unrest or a protest event",
            "unrelated news or general discussion"
        ]
    )

    detected_label = detection["labels"][0]
    detection_score = float(detection["scores"][0])

    # Conservative filtering threshold (needs validation)
    if (
        detected_label != "text reporting civil unrest or a protest event"
        or detection_score < 0.70
    ):
        return {
            "text": text,
            "event_detected": False,
            "confidence": round(detection_score, 4),
            "event_type": None,
            "locations": [],
            "organizations": [],
            "dates": []
        }

    # Stage 2: Event type classification
    event_result = event_classifier(
        text,
        candidate_labels=EVENT_LABELS
    )

    doc = nlp(text)

    return {
        "text": text,
        "event_detected": True,
        "confidence": round(detection_score, 4),
        "event_type": event_result["labels"][0],
        "event_type_confidence": round(
            float(event_result["scores"][0]), 4
        ),
        "locations": list(set(
            ent.text for ent in doc.ents
            if ent.label_ in ["GPE", "LOC"]
        )),
        "organizations": list(set(
            ent.text for ent in doc.ents
            if ent.label_ == "ORG"
        )),
        "dates": list(set(
            ent.text for ent in doc.ents
            if ent.label_ in ["DATE", "TIME"]
        ))
    }


if __name__ == "__main__":

    texts = [
        "Workers protested in Delhi on Monday over unpaid wages.",
        "Students organized a march in Kolkata against rising tuition fees.",
        "India won the cricket match yesterday.",
        "The weather is pleasant today.",
        "Thousands gathered to demand better working conditions."
    ]

    for text in texts:
        print("\n", extract_event(text))