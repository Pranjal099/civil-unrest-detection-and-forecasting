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

# Protest-related keywords for trigger detection
PROTEST_KEYWORDS = {
    "protest", "protests", "protested", "protesting", "protester", "protesters",
    "strike", "strikes", "struck", "striking",
    "march", "marches", "marched", "marching",
    "rally", "rallies", "rallied", "rallying",
    "demonstration", "demonstrations", "demonstrate", "demonstrated", "demonstrating",
    "riot", "riots", "rioted", "rioting",
    "agitation", "agitating", "agitate",
    "blockade", "blockades", "blocked", "blocking", "roadblock", "chakka jam",
    "boycott", "boycotts", "boycotting",
    "uprising", "revolt", "picket", "picketing",
    "walkout", "dharna", "bandh", "hartal", "gherao", "mob", "unrest",
    "clash", "clashes", "clashed", "clashing"
}

# Crowd/scale adjacent keywords
CROWD_KEYWORDS = {
    "crowd", "people", "worker", "workers", "farmer", "farmers",
    "protesters", "demonstrators", "students", "citizens", "supporters",
    "activists", "teachers"
}

# Escalation indicators
ESCALATION_INDICATORS = [
    "violence", "clash", "arrest", "teargas", "lathi", "pellet",
    "injured", "killed", "blocked", "detained", "vandalism", "arson",
    "strike", "riot", "stone pelting", "water cannon", "curfew", "barricade", "chakka jam"
]

# Mobilization call patterns
MOBILIZATION_PATTERNS = [
    "join us", "march with", "rally at", "support farmers",
    "come out", "protest at", "gather at", "stand with", "rise up",
    "take to the streets", "solidarity with", "call to action",
    "we demand", "join the", "be there", "show up", "call for",
    "assemble at", "reach delhi", "chalo"
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


def _extract_spacy_fields(doc, locations: List[str], raw_text: str = "") -> Dict[str, Any]:  # type: ignore
    """
    Extract trigger_phrase, actors, target, scale from a spaCy doc.

    Parameters
    ----------
    doc : spacy.tokens.Doc
        Parsed spaCy document.
    locations : list of str
        Already-extracted location strings (to avoid duplicating as target).
    raw_text : str, optional
        Original raw text (used as fallback when doc is a mock or lacks text attribute).

    Returns
    -------
    dict with keys: trigger_phrase, actors, target, scale
    """
    trigger_phrase: Optional[str] = None
    actors: List[str] = []
    target: Optional[str] = None
    scale: Optional[str] = None

    try:
        # --- Trigger phrase: find root verb or noun near protest keywords ---
        trigger_token = None
        text_lower = doc.text.lower()

        # Walk tokens looking for a protest-related keyword
        for token in doc:
            lemma = token.lemma_.lower()
            tok_text = token.text.lower()
            if lemma in PROTEST_KEYWORDS or tok_text in PROTEST_KEYWORDS:
                # Prefer the actual keyword token as trigger anchor
                if token.pos_ in ("VERB", "NOUN"):
                    trigger_token = token
                    break

        # Fallback: use the sentence root verb if it touches a protest keyword
        if trigger_token is None:
            for sent in doc.sents:
                if sent.root.pos_ == "VERB":
                    sent_text = sent.text.lower()
                    if any(kw in sent_text for kw in PROTEST_KEYWORDS):
                        trigger_token = sent.root
                        break

        if trigger_token is not None:
            # Build trigger phrase: verb + direct object if available
            dobj_texts = [
                child.text for child in trigger_token.children
                if child.dep_ in ("dobj", "pobj", "attr")
            ]
            if dobj_texts:
                trigger_phrase = f"{trigger_token.lemma_} {dobj_texts[0]}"
            else:
                trigger_phrase = trigger_token.lemma_

            # --- Actors: PERSON/ORG that are nsubj/agent of trigger_token ---
            subject_ents: List[str] = []
            for child in trigger_token.children:
                if child.dep_ in ("nsubj", "nsubjpass", "agent", "csubj"):
                    # Find any named entity that spans this child
                    for ent in doc.ents:
                        if ent.label_ in ("PERSON", "ORG"):
                            if any(t.i == child.i or (child.i <= t.i <= child.i) for t in ent):
                                subject_ents.append(ent.text)

            # If nothing found via dependency, fall back to all PERSON ents
            if not subject_ents:
                subject_ents = [
                    ent.text for ent in doc.ents
                    if ent.label_ in ("PERSON", "ORG")
                ]

            actors = list(dict.fromkeys(subject_ents))[:5]  # deduplicate, max 5

            # --- Target: GPE/ORG that is dobj/pobj of trigger, not in locations ---
            location_set = set(loc.lower() for loc in locations)
            for child in trigger_token.children:
                if child.dep_ in ("dobj", "pobj", "nmod", "prep"):
                    for ent in doc.ents:
                        if ent.label_ in ("GPE", "ORG"):
                            if any(t.i == child.i for t in ent):
                                if ent.text.lower() not in location_set:
                                    target = ent.text
                                    break
                if target:
                    break

        # --- Scale: CARDINAL entity adjacent to crowd/people keywords ---
        tokens_list = list(doc)
        for i, token in enumerate(tokens_list):
            if token.ent_type_ == "CARDINAL":
                # Check window of ±3 tokens for crowd keywords
                window_start = max(0, i - 3)
                window_end = min(len(tokens_list), i + 4)
                window_texts = {tokens_list[j].text.lower() for j in range(window_start, window_end)}
                if window_texts & CROWD_KEYWORDS:
                    # Find the full CARDINAL entity text
                    for ent in doc.ents:
                        if ent.label_ == "CARDINAL" and any(t.i == token.i for t in ent):
                            scale = ent.text
                            break
                    if scale:
                        break

    except Exception as exc:  # type: ignore
        logger.debug("spaCy field extraction error: %s", exc)

    # --- Robust fallbacks (works on both full docs and mock objects) ---
    doc_text = getattr(doc, "text", "") or raw_text or ""
    import re

    # Fallback actors from ents if still empty
    if not actors and hasattr(doc, "ents"):
        actors = list(dict.fromkeys([
            ent.text for ent in doc.ents
            if getattr(ent, "label_", "") in ("PERSON", "ORG", "NORP")
        ]))[:5]

    # Fallback trigger phrase via regex
    if not trigger_phrase and doc_text:
        for kw in sorted(PROTEST_KEYWORDS, key=len, reverse=True):
            m = re.search(rf"\b({kw}(?:\s+[a-z]+)?)\b", doc_text, re.IGNORECASE)
            if m:
                trigger_phrase = m.group(1).strip()
                break

    # Fallback target via prepositional regex
    if not target and doc_text:
        m = re.search(r"\b(?:against|demanding|opposing|over|to repeal)\s+([A-Za-z0-9\s\-']{3,40}?)(?=[,.;]|\bin\b|\bon\b|\bat\b|$)", doc_text, re.IGNORECASE)
        if m:
            cand = m.group(1).strip()
            loc_set = set(loc.lower() for loc in locations)
            if cand.lower() not in loc_set:
                target = cand

    # Fallback scale via regex
    if not scale and doc_text:
        m = re.search(r"\b(\d+[\d,]*\+?|\bthousands?\b|\bhundreds?\b|\bmillions?\b|\bmassive\b|\blarge\b)\s+(?:of\s+)?(?:people|protesters|demonstrators|farmers|workers|students|crowd|citizens)\b", doc_text, re.IGNORECASE)
        if m:
            scale = m.group(0).strip()
        elif "massive" in doc_text.lower():
            scale = "massive"

    return {
        "trigger_phrase": trigger_phrase,
        "actors": actors,
        "target": target,
        "scale": scale,
    }


def extract_event(
    text: Any,
    threshold: float = 0.70,
    detector_pipeline=None,
    classifier_pipeline=None,
    nlp_model=None,
    source: Optional[str] = None,
    timestamp: Optional[str] = None,
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
    source : str, optional
        Pass-through provenance field (set by caller).
    timestamp : str, optional
        Pass-through timestamp field (set by caller).

    Returns
    -------
    dict
        Structured extraction results with event detection flag, category,
        confidence scores, and extracted entity lists (locations, orgs, dates),
        plus trigger_phrase, actors, target, scale, escalation_indicators,
        mobilization_call, source, timestamp.
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
            "dates": [],
            "trigger_phrase": None,
            "actors": [],
            "target": None,
            "scale": None,
            "escalation_indicators": [],
            "mobilization_call": False,
            "source": source,
            "timestamp": timestamp,
        }

    clean_text = text.strip()
    text_lower = clean_text.lower()
    detector = detector_pipeline if detector_pipeline is not None else get_detector()

    # --- Escalation indicators (purely string-based, always computed) ---
    escalation_indicators: List[str] = [
        kw for kw in ESCALATION_INDICATORS if kw in text_lower
    ]

    # --- Mobilization call (purely string-based, always computed) ---
    mobilization_call: bool = any(pat in text_lower for pat in MOBILIZATION_PATTERNS)

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
            "dates": [],
            "trigger_phrase": None,
            "actors": [],
            "target": None,
            "scale": None,
            "escalation_indicators": escalation_indicators,
            "mobilization_call": mobilization_call,
            "source": source,
            "timestamp": timestamp,
        }

    # Stage 2: Event typology classification & Named Entity Recognition
    classifier = classifier_pipeline if classifier_pipeline is not None else get_classifier()
    nlp = nlp_model if nlp_model is not None else get_nlp()

    event_result = classifier(clean_text, candidate_labels=EVENT_LABELS)
    doc = nlp(clean_text)

    locations = sorted(list(set(
        ent.text for ent in doc.ents
        if ent.label_ in ["GPE", "LOC"]
    )))
    organizations = sorted(list(set(
        ent.text for ent in doc.ents
        if ent.label_ == "ORG"
    )))
    dates = sorted(list(set(
        ent.text for ent in doc.ents
        if ent.label_ in ["DATE", "TIME"]
    )))

    # SpaCy-based structural fields
    spacy_fields = _extract_spacy_fields(doc, locations, raw_text=clean_text)

    return {
        "text": clean_text,
        "event_detected": True,
        "confidence": round(detection_score, 4),
        "event_type": event_result["labels"][0],
        "event_type_confidence": round(float(event_result["scores"][0]), 4),
        "locations": locations,
        "organizations": organizations,
        "dates": dates,
        "trigger_phrase": spacy_fields["trigger_phrase"],
        "actors": spacy_fields["actors"],
        "target": spacy_fields["target"],
        "scale": spacy_fields["scale"],
        "escalation_indicators": escalation_indicators,
        "mobilization_call": mobilization_call,
        "source": source,
        "timestamp": timestamp,
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
