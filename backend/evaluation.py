from typing import Any, Dict, Union
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    average_precision_score,
    confusion_matrix
)


def evaluate_model(
    y_true: Union[np.ndarray, pd.Series, list],
    y_prob: Union[np.ndarray, pd.Series, list],
    threshold: float = 0.5
) -> Dict[str, Any]:
    """
    Evaluate a binary classification model against probability forecasts.

    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        True binary class labels (0 or 1).
    y_prob : array-like of shape (n_samples,)
        Predicted positive-class probabilities in [0.0, 1.0].
    threshold : float, default 0.5
        Decision boundary for binary classification.

    Returns
    -------
    dict
        Dictionary containing threshold, accuracy, precision, recall, f1_score,
        pr_auc, balanced_accuracy, positive_prevalence, and confusion matrix.
    """
    y_true_arr = np.asarray(y_true, dtype=int)
    y_prob_arr = np.asarray(y_prob, dtype=float)

    if len(y_true_arr) != len(y_prob_arr):
        raise ValueError(
            f"Length mismatch: y_true has {len(y_true_arr)} samples, "
            f"y_prob has {len(y_prob_arr)} samples."
        )

    if len(y_true_arr) == 0:
        raise ValueError("Cannot evaluate empty predictions array.")

    # Convert continuous probabilities into binary predictions
    y_pred = (y_prob_arr >= threshold).astype(int)

    # Core performance metrics
    acc = float(accuracy_score(y_true_arr, y_pred))
    prec = float(precision_score(y_true_arr, y_pred, zero_division=0))
    rec = float(recall_score(y_true_arr, y_pred, zero_division=0))
    f1 = float(f1_score(y_true_arr, y_pred, zero_division=0))
    pr_auc = float(average_precision_score(y_true_arr, y_prob_arr))
    bal_acc = float(balanced_accuracy_score(y_true_arr, y_pred))
    cm = confusion_matrix(y_true_arr, y_pred).tolist()

    results = {
        "threshold": round(float(threshold), 4),
        "accuracy": round(acc, 4),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1_score": round(f1, 4),
        "pr_auc": round(pr_auc, 4),
        "balanced_accuracy": round(bal_acc, 4),
        "positive_prevalence": round(float(y_true_arr.mean()), 4),
        "confusion_matrix": cm
    }

    return results


def format_metrics_summary(results: Dict[str, Any], title: str = "Model Evaluation") -> str:
    """Format evaluation metrics as an aligned ASCII report."""
    lines = [
        "=" * 45,
        f" {title.upper()}",
        "=" * 45,
        f" Decision Threshold   : {results.get('threshold', 0.5):.2f}",
        f" Accuracy             : {results.get('accuracy', 0.0):.4f}",
        f" Precision            : {results.get('precision', 0.0):.4f}",
        f" Recall               : {results.get('recall', 0.0):.4f}",
        f" F1 Score             : {results.get('f1_score', 0.0):.4f}",
        f" PR-AUC               : {results.get('pr_auc', 0.0):.4f}",
        f" Balanced Accuracy    : {results.get('balanced_accuracy', 0.0):.4f}",
        f" Positive Prevalence  : {results.get('positive_prevalence', 0.0):.2%}",
        "-" * 45,
        f" Confusion Matrix     : {results.get('confusion_matrix')}",
        "=" * 45
    ]
    return "\n".join(lines)
