from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    average_precision_score,
    confusion_matrix
)


def evaluate_model(y_true, y_prob, threshold=0.5):
    """
    Evaluate a binary classification model.
    """

    # Convert probabilities into binary predictions
    y_pred = (y_prob >= threshold).astype(int)

    # Calculate metrics
    results = {
        "threshold": threshold,
        "accuracy": round(accuracy_score(y_true, y_pred), 4),
        "precision": round(precision_score(y_true, y_pred, zero_division=0), 4),
        "recall": round(recall_score(y_true, y_pred, zero_division=0), 4),
        "f1_score": round(f1_score(y_true, y_pred, zero_division=0), 4),
        "pr_auc": round(average_precision_score(y_true, y_prob), 4),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist()
    }

    return results