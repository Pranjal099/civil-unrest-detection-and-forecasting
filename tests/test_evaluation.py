import pytest
import numpy as np
import pandas as pd

from backend.evaluation import evaluate_model, format_metrics_summary


def test_evaluate_model_perfect_predictions():
    y_true = [0, 0, 1, 1]
    y_prob = [0.1, 0.2, 0.8, 0.9]

    metrics = evaluate_model(y_true, y_prob, threshold=0.5)

    assert metrics["accuracy"] == 1.0
    assert metrics["precision"] == 1.0
    assert metrics["recall"] == 1.0
    assert metrics["f1_score"] == 1.0
    assert metrics["pr_auc"] == 1.0
    assert metrics["balanced_accuracy"] == 1.0
    assert metrics["positive_prevalence"] == 0.5
    assert metrics["confusion_matrix"] == [[2, 0], [0, 2]]


def test_evaluate_model_threshold_variation():
    y_true = [0, 0, 1, 1]
    y_prob = [0.2, 0.4, 0.6, 0.8]

    # At threshold 0.3, [0.4, 0.6, 0.8] are predicted positive -> recall 1.0, precision 2/3
    low_thresh = evaluate_model(y_true, y_prob, threshold=0.3)
    assert low_thresh["recall"] == 1.0
    assert round(low_thresh["precision"], 2) == 0.67

    # At threshold 0.7, only [0.8] is predicted positive -> recall 0.5, precision 1.0
    high_thresh = evaluate_model(y_true, y_prob, threshold=0.7)
    assert high_thresh["recall"] == 0.5
    assert high_thresh["precision"] == 1.0


def test_evaluate_model_zero_division():
    """Verify that all-negative predictions do not cause ZeroDivisionError."""
    y_true = [0, 0, 1, 1]
    y_prob = [0.1, 0.1, 0.1, 0.1]  # all below threshold

    metrics = evaluate_model(y_true, y_prob, threshold=0.5)
    assert metrics["precision"] == 0.0
    assert metrics["recall"] == 0.0
    assert metrics["f1_score"] == 0.0
    assert metrics["accuracy"] == 0.5


def test_evaluate_model_input_types():
    """Verify compatibility with pandas Series, numpy arrays, and Python lists."""
    y_true_s = pd.Series([1, 0, 1, 0])
    y_prob_a = np.array([0.9, 0.2, 0.7, 0.1])

    metrics = evaluate_model(y_true_s, y_prob_a, threshold=0.5)
    assert metrics["accuracy"] == 1.0


def test_evaluate_model_mismatched_lengths():
    with pytest.raises(ValueError, match="Length mismatch"):
        evaluate_model([1, 0], [0.5, 0.6, 0.7])


def test_evaluate_model_empty_input():
    with pytest.raises(ValueError, match="empty predictions array"):
        evaluate_model([], [])


def test_format_metrics_summary():
    y_true = [0, 1, 1, 0]
    y_prob = [0.1, 0.9, 0.8, 0.2]
    metrics = evaluate_model(y_true, y_prob, threshold=0.5)

    summary = format_metrics_summary(metrics, title="2024 Test Set")
    assert "2024 TEST SET" in summary
    assert "Accuracy" in summary
    assert "Confusion Matrix" in summary
