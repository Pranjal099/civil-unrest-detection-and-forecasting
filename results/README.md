# Evaluation Results & Benchmark Summary

This directory stores evaluation metrics, benchmark artifacts, and event extraction datasets generated during model training, evaluation, and case study analysis.

---

## 1. 2024 Held-Out Test Set Performance

The multi-horizon XGBoost models were evaluated on the complete 2024 calendar year (19,386 state-day observations) strictly held out from training:

| Horizon | Decision Threshold | PR-AUC | F1-Score | Precision | Recall | Accuracy |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **1-Day** | 0.35 | **0.9194** | 0.8398 | 0.8144 | 0.8670 | 80.60% |
| **3-Day** | 0.25 | **0.9757** | 0.9161 | 0.8838 | 0.9509 | 86.45% |
| **7-Day** | 0.10 | **0.9944** | 0.9681 | 0.9502 | 0.9868 | 94.15% |

### Key Observations
* **High PR-AUC across all horizons**: Area under Precision-Recall curve reaches 0.9194 for 1-day, 0.9757 for 3-day, and 0.9944 for 7-day, demonstrating strong discriminatory power over minority positive events.
* **Calibrated Thresholds**: Decision thresholds were selected during validation tuning to maximize recall without sacrificing precision on imbalanced event occurrences.
* **Recall Prioritization**: In civil unrest early warning contexts, false negatives carry high risk; recall exceeds 86% across all horizons and reaches 98.7% for the 7-day horizon.

---

## 2. Artifacts Catalog

| File | Description | Records | Source Pipeline |
| :--- | :--- | :--- | :--- |
| `final_results.csv` | Official test metrics (PR-AUC, F1, Precision, Recall, Accuracy) across 1, 3, 7 days | 3 horizons | `notebooks/Untitled74-2.ipynb`, `backend/evaluate_dataset.py` |
| `demo_extracted_events.csv` | NLP zero-shot extractions and entity recognitions for demonstration posts | 10 posts | `backend/process_social_posts.py` |
| `final_protest_extractions.csv` | Verified protest extractions within historical event windows | 47 articles | `backend/process_protest_articles.py` |
| `protest_extracted_events.csv` | Full set of protest article extractions from RSS queries | 120 articles | `backend/social_media_extraction.py` |
| `protest_timeline.csv` | Aggregated temporal frequency timeline by protest event | 85 dates | `backend/analyze_protest_timeline.py` |

---

## 3. Reproduction Instructions

To re-run the 2024 test set evaluation against the persisted XGBoost models:

```bash
make evaluate
# Or directly via Python:
python backend/evaluate_dataset.py
```
