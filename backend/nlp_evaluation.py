"""
nlp_evaluation.py
-----------------
Loads benchmark model predictions and reports model-side statistics.
If human annotations are present, computes precision/recall/F1 for event detection.

Usage:
    python backend/nlp_evaluation.py [--predictions results/benchmark_model_predictions.csv]
"""

import argparse
import ast
import sys
from pathlib import Path

import pandas as pd


def _truthy_list(value) -> bool:
    """Return True if the cell contains a non-empty list (even as string repr)."""
    if value is None:
        return False
    if isinstance(value, list):
        return len(value) > 0
    if isinstance(value, str):
        stripped = value.strip()
        if stripped in ("", "[]", "None"):
            return False
        try:
            parsed = ast.literal_eval(stripped)
            return isinstance(parsed, list) and len(parsed) > 0
        except (ValueError, SyntaxError):
            return stripped not in ("", "None")
    return bool(value)


def report_model_statistics(df: pd.DataFrame) -> None:
    """Print model-output statistics to stdout."""
    n = len(df)
    print("=" * 60)
    print("BENCHMARK MODEL-OUTPUT STATISTICS")
    print("=" * 60)
    print(f"  Total posts analyzed          : {n}")

    if "event_detected_model" in df.columns:
        n_detected = df["event_detected_model"].sum()
        pct = 100 * n_detected / n if n else 0.0
        print(f"  Detected as civil unrest      : {int(n_detected)} / {n}  ({pct:.1f}%)")
    else:
        print("  [WARNING] 'event_detected_model' column not found")

    if "event_type_model" in df.columns:
        print("\n  Event type distribution:")
        counts = df["event_type_model"].value_counts(dropna=False)
        for label, cnt in counts.items():
            print(f"    {str(label):<25} {cnt:>5}  ({100*cnt/n:.1f}%)")

    if "trigger_phrase_model" in df.columns:
        n_trigger = df["trigger_phrase_model"].apply(
            lambda v: v not in (None, "", "None") if not isinstance(v, float) else False
        ).sum()
        print(f"\n  With trigger phrase           : {int(n_trigger)} / {n}  ({100*n_trigger/n:.1f}%)")

    if "actors_model" in df.columns:
        n_actors = df["actors_model"].apply(_truthy_list).sum()
        print(f"  With actors identified        : {int(n_actors)} / {n}  ({100*n_actors/n:.1f}%)")

    if "locations_model" in df.columns:
        n_locs = df["locations_model"].apply(_truthy_list).sum()
        print(f"  With locations found          : {int(n_locs)} / {n}  ({100*n_locs/n:.1f}%)")

    if "escalation_indicators_model" in df.columns:
        n_esc = df["escalation_indicators_model"].apply(_truthy_list).sum()
        print(f"  With escalation indicators    : {int(n_esc)} / {n}  ({100*n_esc/n:.1f}%)")

    if "mobilization_call_model" in df.columns:
        n_mob = df["mobilization_call_model"].sum()
        print(f"  With mobilization call        : {int(n_mob)} / {n}  ({100*n_mob/n:.1f}%)")

    print()
    print(
        "NOTE: Human annotations not yet collected. "
        "These are model-output statistics only, not evaluation metrics."
    )
    print("=" * 60)


def compute_evaluation_metrics(df: pd.DataFrame) -> None:
    """Compute precision/recall/F1 if human labels are present."""
    if "event_detected_human" not in df.columns:
        return

    human_col = df["event_detected_human"]
    # Only evaluate rows where human label is provided (non-null, non-empty)
    labeled_mask = human_col.notna() & (human_col.astype(str).str.strip() != "")
    if labeled_mask.sum() == 0:
        print("\nHuman annotations column present but empty — skipping evaluation metrics.")
        return

    labeled = df[labeled_mask].copy()
    human_labels = labeled["event_detected_human"].astype(int)

    if "event_detected_model" not in labeled.columns:
        print("\n[ERROR] 'event_detected_model' column missing — cannot compute metrics.")
        return

    model_preds = labeled["event_detected_model"].astype(int)

    tp = int(((model_preds == 1) & (human_labels == 1)).sum())
    fp = int(((model_preds == 1) & (human_labels == 0)).sum())
    fn = int(((model_preds == 0) & (human_labels == 1)).sum())

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    n_labeled = int(labeled_mask.sum())
    print(f"\n{'=' * 60}")
    print("EVALUATION METRICS  (vs human annotations)")
    print(f"{'=' * 60}")
    print(f"  Labeled rows evaluated        : {n_labeled}")
    print(f"  True Positives                : {tp}")
    print(f"  False Positives               : {fp}")
    print(f"  False Negatives               : {fn}")
    print(f"  Precision                     : {precision:.4f}")
    print(f"  Recall                        : {recall:.4f}")
    print(f"  F1 Score                      : {f1:.4f}")
    print("=" * 60)


def main(predictions_path: str = "results/benchmark_model_predictions.csv") -> None:
    path = Path(predictions_path)
    if not path.exists():
        print(f"[ERROR] Predictions file not found: {path}", file=sys.stderr)
        sys.exit(1)

    df = pd.read_csv(path)
    report_model_statistics(df)
    compute_evaluation_metrics(df)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate NLP benchmark predictions")
    parser.add_argument(
        "--predictions",
        default="results/benchmark_model_predictions.csv",
        help="Path to model predictions CSV"
    )
    args = parser.parse_args()
    main(args.predictions)
