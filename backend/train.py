import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, Optional, Tuple, Any

# Ensure project root is in sys.path for direct script execution
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    accuracy_score
)

from backend.data_processing import FEATURES, load_and_process_data

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

ROOT_DIR = Path(__file__).resolve().parent.parent
DEFAULT_THRESHOLDS = {1: 0.35, 3: 0.25, 7: 0.10}


def prepare_temporal_splits(
    data: pd.DataFrame,
    features: list,
    target_col: str,
    train_end_year: int = 2022,
    val_year: int = 2023,
    test_year: int = 2024
) -> Dict[str, Tuple[pd.DataFrame, pd.Series]]:
    """
    Split time-series data chronologically to prevent lookahead leakage.
    """
    train_mask = data["date"].dt.year <= train_end_year
    val_mask = data["date"].dt.year == val_year
    test_mask = data["date"].dt.year == test_year

    splits = {
        "train": (data.loc[train_mask, features], data.loc[train_mask, target_col].astype(int)),
        "val": (data.loc[val_mask, features], data.loc[val_mask, target_col].astype(int)),
        "test": (data.loc[test_mask, features], data.loc[test_mask, target_col].astype(int))
    }
    return splits


def train_single_horizon(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    random_state: int = 42,
    n_estimators: int = 300,
    max_depth: int = 5,
    learning_rate: float = 0.05
) -> XGBClassifier:
    """
    Fit an XGBoost classifier with inverse class-frequency scaling.
    """
    pos_count = max(int((y_train == 1).sum()), 1)
    neg_count = int((y_train == 0).sum())
    scale_pos_weight = neg_count / pos_count

    model = XGBClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight,
        eval_metric="logloss",
        random_state=random_state,
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    return model


def evaluate_horizon_model(
    model: XGBClassifier,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    threshold: float
) -> Dict[str, float]:
    """
    Compute classification performance metrics on held-out test data.
    """
    probs = model.predict_proba(X_test)[:, 1]
    preds = (probs >= threshold).astype(int)

    return {
        "pr_auc": float(average_precision_score(y_test, probs)),
        "f1": float(f1_score(y_test, preds, zero_division=0)),
        "precision": float(precision_score(y_test, preds, zero_division=0)),
        "recall": float(recall_score(y_test, preds, zero_division=0)),
        "accuracy": float(accuracy_score(y_test, preds))
    }


def run_training_pipeline(
    data: Optional[pd.DataFrame] = None,
    output_dir: Optional[Path] = None,
    dry_run: bool = False,
    horizons: Optional[list] = None
) -> Dict[str, Any]:
    """
    Orchestrate multi-horizon model training and evaluation.
    Safely runs in dry-run mode without mutating existing production assets.
    """
    if horizons is None:
        horizons = [1, 3, 7]

    if data is None:
        logger.info("Loading time-series dataset...")
        data = load_and_process_data()

    logger.info("Dataset shape for training: %s", data.shape)
    results = {}

    for horizon in horizons:
        target_col = f"target_{horizon}day"
        threshold = DEFAULT_THRESHOLDS.get(horizon, 0.5)

        logger.info("Processing horizon %d-day (target: %s)...", horizon, target_col)
        splits = prepare_temporal_splits(data, FEATURES, target_col)

        X_train, y_train = splits["train"]
        X_val, y_val = splits["val"]
        X_test, y_test = splits["test"]

        # Combine train and val for final fit
        X_comb = pd.concat([X_train, X_val], ignore_index=True)
        y_comb = pd.concat([y_train, y_val], ignore_index=True)

        if dry_run:
            logger.info("Dry-run mode: verified split shapes. Train: %s, Test: %s", X_comb.shape, X_test.shape)
            results[horizon] = {
                "train_samples": len(X_comb),
                "test_samples": len(X_test),
                "pos_prevalence": float(y_test.mean())
            }
            continue

        model = train_single_horizon(X_comb, y_comb)
        metrics = evaluate_horizon_model(model, X_test, y_test, threshold)
        results[horizon] = metrics
        logger.info("Horizon %d-day test PR-AUC: %.4f | F1: %.4f", horizon, metrics["pr_auc"], metrics["f1"])

        if output_dir:
            output_dir.mkdir(parents=True, exist_ok=True)
            model_file = output_dir / f"xgboost_{horizon}day.json"
            model.save_model(str(model_file))
            logger.info("Saved model to %s", model_file)

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Civil Unrest XGBoost Training Pipeline")
    parser.add_argument("--dry-run", action="store_true", help="Validate splits without fitting or saving models")
    parser.add_argument("--output-dir", type=str, default=None, help="Directory to save newly trained models")
    args = parser.parse_args()

    out_path = Path(args.output_dir) if args.output_dir else None
    run_training_pipeline(dry_run=args.dry_run, output_dir=out_path)
