import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

df = pd.read_csv("results/demo_extracted_events.csv")

# Ground-truth labels for the 10 synthetic demo posts
y_true = [1, 1, 1, 0, 0, 1, 0, 1, 0, 1]
y_pred = df["event_detected"].astype(int)

print("Demo NLP Evaluation")
print("-" * 30)
print("Accuracy :", accuracy_score(y_true, y_pred))
print("Precision:", precision_score(y_true, y_pred))
print("Recall   :", recall_score(y_true, y_pred))
print("F1 Score :", f1_score(y_true, y_pred))
