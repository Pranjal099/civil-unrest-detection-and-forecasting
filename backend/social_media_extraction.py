import logging
from typing import Any, Dict, List, Optional

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

EVENT_LABELS: List[str] = [
    "protest",
    "demonstration",
    "strike",
    "riot",
    "labor dispute",
    "political unrest",
    "other"
]

CANDIDATE_DETECTION_LABELS = [
    "text reporting civil unrest or a protest event",
    "unrelated news or general discussion"
]

# Shared pipeline caches
_nlp = None
_detector = None
_classifier = None


def get_device() -> str:
    """Detect and return optimal compute accelerator device."""
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
    except ImportError:
        pass
    return "cpu"


def get_nlp():
    """Lazily load and cache SpaCy linguistic model."""
    global _nlp
    if _nlp is None:
        import spacy
        try:
            _nlp = spacy.load("en_core_web_sm")
        except OSError:
            logger.warning("en_core_web_sm not found, downloading...")
            from spacy.cli import download
            download("en_core_web_sm")
            _nlp = spacy.load("en_core_web_sm")
    return _nlp


def get_detector():
    """Lazily load and cache zero-shot civil unrest binary detector."""
    global _detector
    if _detector is None:
        from transformers import pipeline
        device = get_device()
        logger.info("Initializing zero-shot unrest detector on device: %s", device)
        device_arg = 0 if device == "cuda" else (device if device == "mps" else -1)
        _detector = pipeline(
            "zero-shot-classification",
            model="facebook/bart-large-mnli",
            device=device_arg
        )
    return _detector


def get_classifier():
    """Lazily load and cache zero-shot event typology classifier."""
    global _classifier
    if _classifier is None:
        from transformers import pipeline
        device = get_device()
        logger.info("Initializing zero-shot typology classifier on device: %s", device)
        device_arg = 0 if device == "cuda" else (device if device == "mps" else -1)
        _classifier = pipeline(
            "zero-shot-classification",
            model="facebook/bart-large-mnli",
            device=device_arg
        )
    return _classifier


def extract_event(
    text: Any,
    threshold: float = 0.70,
    detector_pipeline=None,
    classifier_pipeline=None,
    nlp_model=None
) -> Dict[str, Any]:
    """
    Extract civil unrest signals, classify event category, and extract entities.

    Parameters
    ----------
    text : str
        Input social media post or headline text.
    threshold : float, default 0.70
        Confidence score threshold required for positive unrest detection.
    detector_pipeline : optional
        Override for binary detection pipeline (useful for testing).
    classifier_pipeline : optional
        Override for event typology classification pipeline.
    nlp_model : optional
        Override for SpaCy NER pipeline.

    Returns
    -------
    dict
        Structured extraction results with event detection flag, category,
        confidence scores, and extracted entity lists (locations, orgs, dates).
    """
    # Defensive input sanitization
    if text is None or not isinstance(text, str) or not text.strip():
        return {
            "text": "" if text is None else str(text),
            "event_detected": False,
            "confidence": 0.0,
            "event_type": None,
            "locations": [],
            "organizations": [],
            "dates": []
        }

    clean_text = text.strip()
    detector = detector_pipeline if detector_pipeline is not None else get_detector()

    # Stage 1: Binary unrest classification
    detection = detector(
        clean_text,
        candidate_labels=CANDIDATE_DETECTION_LABELS
    )

    detected_label = detection["labels"][0]
    detection_score = float(detection["scores"][0])

    if (
        detected_label != "text reporting civil unrest or a protest event"
        or detection_score < threshold
    ):
        return {
            "text": clean_text,
            "event_detected": False,
            "confidence": round(detection_score, 4),
            "event_type": None,
            "locations": [],
            "organizations": [],
            "dates": []
        }

    # Stage 2: Event typology classification & Named Entity Recognition
    classifier = classifier_pipeline if classifier_pipeline is not None else get_classifier()
    nlp = nlp_model if nlp_model is not None else get_nlp()

    event_result = classifier(clean_text, candidate_labels=EVENT_LABELS)
    doc = nlp(clean_text)

    return {
        "text": clean_text,
        "event_detected": True,
        "confidence": round(detection_score, 4),
        "event_type": event_result["labels"][0],
        "event_type_confidence": round(float(event_result["scores"][0]), 4),
        "locations": sorted(list(set(
            ent.text for ent in doc.ents
            if ent.label_ in ["GPE", "LOC"]
        ))),
        "organizations": sorted(list(set(
            ent.text for ent in doc.ents
            if ent.label_ == "ORG"
        ))),
        "dates": sorted(list(set(
            ent.text for ent in doc.ents
            if ent.label_ in ["DATE", "TIME"]
        )))
    }


if __name__ == "__main__":
    sample_texts = [
        "Workers protested in Delhi on Monday over unpaid wages.",
        "Students organized a march in Kolkata against rising tuition fees.",
        "India won the cricket match yesterday.",
        "The weather is pleasant today.",
        "Thousands gathered to demand better working conditions."
    ]

    for t in sample_texts:
        print("\n", extract_event(t))
