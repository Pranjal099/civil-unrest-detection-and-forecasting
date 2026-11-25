import pytest
from unittest.mock import MagicMock

from backend.social_media_extraction import (
    EVENT_LABELS,
    extract_event,
    get_device
)


def test_device_detection():
    device = get_device()
    assert device in {"cuda", "mps", "cpu"}


def test_input_sanitization_empty():
    for invalid_input in ["", "   ", "\n\t", None]:
        res = extract_event(invalid_input)
        assert res["event_detected"] is False
        assert res["confidence"] == 0.0
        assert res["event_type"] is None
        assert res["locations"] == []
        assert res["organizations"] == []
        assert res["dates"] == []


def test_input_sanitization_non_string():
    res = extract_event(12345)
    assert res["event_detected"] is False
    assert res["confidence"] == 0.0


def test_extract_event_below_threshold_with_mock():
    mock_detector = MagicMock(return_value={
        "labels": ["text reporting civil unrest or a protest event", "unrelated news"],
        "scores": [0.65, 0.35]
    })

    res = extract_event(
        "Some ambiguous news headline",
        threshold=0.70,
        detector_pipeline=mock_detector
    )
    assert res["event_detected"] is False
    assert res["confidence"] == 0.65
    assert res["event_type"] is None


def test_extract_event_unrelated_label_with_mock():
    mock_detector = MagicMock(return_value={
        "labels": ["unrelated news or general discussion", "text reporting civil unrest"],
        "scores": [0.95, 0.05]
    })

    res = extract_event(
        "Cricket match score update",
        threshold=0.70,
        detector_pipeline=mock_detector
    )
    assert res["event_detected"] is False
    assert res["confidence"] == 0.95
    assert res["event_type"] is None


def test_extract_event_positive_with_mocks():
    mock_detector = MagicMock(return_value={
        "labels": ["text reporting civil unrest or a protest event", "unrelated news"],
        "scores": [0.92, 0.08]
    })
    mock_classifier = MagicMock(return_value={
        "labels": ["strike", "protest", "riot"],
        "scores": [0.75, 0.15, 0.10]
    })

    # Mock SpaCy Doc and Span entities
    class MockEntity:
        def __init__(self, text, label):
            self.text = text
            self.label_ = label

    class MockDoc:
        def __init__(self, text):
            self.ents = [
                MockEntity("Paris", "LOC"),
                MockEntity("Paris", "GPE"),  # duplicate
                MockEntity("CGT Union", "ORG"),
                MockEntity("Tuesday", "DATE")
            ]

    mock_nlp = MagicMock(side_effect=lambda text: MockDoc(text))

    res = extract_event(
        "Transport workers launched a massive strike in Paris on Tuesday led by CGT Union.",
        threshold=0.70,
        detector_pipeline=mock_detector,
        classifier_pipeline=mock_classifier,
        nlp_model=mock_nlp
    )

    assert res["event_detected"] is True
    assert res["confidence"] == 0.92
    assert res["event_type"] == "strike"
    assert res["event_type_confidence"] == 0.75
    assert res["locations"] == ["Paris"]
    assert res["organizations"] == ["CGT Union"]
    assert res["dates"] == ["Tuesday"]
    assert "CGT Union" in res["actors"]
    assert res["trigger_phrase"] is not None
    assert isinstance(res["escalation_indicators"], list)
    assert "strike" in res["escalation_indicators"]
    assert isinstance(res["mobilization_call"], bool)


def test_extract_event_metadata_passthrough():
    res = extract_event(
        None,
        source="twitter",
        timestamp="2021-02-05T12:00:00Z"
    )
    assert res["source"] == "twitter"
    assert res["timestamp"] == "2021-02-05T12:00:00Z"


def test_extract_event_escalation_and_mobilization():
    mock_detector = MagicMock(return_value={
        "labels": ["text reporting civil unrest or a protest event", "unrelated news"],
        "scores": [0.88, 0.12]
    })
    mock_classifier = MagicMock(return_value={
        "labels": ["protest", "strike"],
        "scores": [0.80, 0.20]
    })

    class MockDoc:
        def __init__(self, text):
            self.ents = []
            self.text = text

    mock_nlp = MagicMock(side_effect=lambda text: MockDoc(text))

    text = "Join us at the border! Police used teargas and lathi charge after violent clash."
    res = extract_event(
        text,
        detector_pipeline=mock_detector,
        classifier_pipeline=mock_classifier,
        nlp_model=mock_nlp
    )
    assert res["event_detected"] is True
    assert res["mobilization_call"] is True
    assert "teargas" in res["escalation_indicators"]
    assert "clash" in res["escalation_indicators"]
    assert "lathi" in res["escalation_indicators"]


def test_event_labels_completeness():
    expected = {"protest", "demonstration", "strike", "riot", "labor dispute", "political unrest", "other"}
    assert set(EVENT_LABELS) == expected
