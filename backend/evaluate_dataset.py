import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    average_precision_score,
    confusion_matrix
)

from backend.data_processing import load_and_process_data, FEATURES
from backend.predict import MODELS, THRESHOLDS


def evaluate_2024():

    data = load_and_process_data()

    # Use 2024 only as the test set
    test_data = data[
        data["date"].dt.year == 2024
    ].copy()

    X_test = test_data[FEATURES]

    print("=" * 60)
    print("CIVIL UNREST FORECAST - 2024 TEST EVALUATION")
    print("=" * 60)

    print("Test records:", len(test_data))
    print("States:", test_data["state"].nunique())

    for horizon in [1, 3, 7]:

        target = f"target_{horizon}day"

        y_true = test_data[target].astype(int)

        # Load model
        model = MODELS[horizon]

        # Predict probabilities
        y_prob = model.predict_proba(X_test)[:, 1]

        # Apply saved threshold
        threshold = float(THRESHOLDS[str(horizon)])
        y_pred = (y_prob >= threshold).astype(int)

        # Metrics
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        pr_auc = average_precision_score(y_true, y_prob)

        cm = confusion_matrix(y_true, y_pred)

        print("\n" + "-" * 60)
        print(f"FORECAST: {horizon} DAY(S)")
        print("-" * 60)

        print(f"Threshold : {threshold}")
        print(f"Accuracy  : {accuracy:.4f}")
        print(f"Precision : {precision:.4f}")
        print(f"Recall    : {recall:.4f}")
        print(f"F1 Score : {f1:.4f}")
        print(f"PR-AUC   : {pr_auc:.4f}")

        print("\nConfusion Matrix:")
        print(cm)

        print("\nActual event prevalence:")
        print(f"{y_true.mean():.2%}")


if __name__ == "__main__":
    evaluate_2024()