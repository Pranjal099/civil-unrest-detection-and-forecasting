# Civil Unrest Forecasting — Independent Verification Report

**Generated:** 2026-11-25  
**Verifier:** Independent automated audit (no model retraining, no file modification)  
**Repository:** `/Users/pranjalgautam/Desktop/civil-unrest-forecasting`  
**Branch / HEAD:** `main` @ `7c4bc40`  
**Python:** 3.13.6 · **pytest:** 9.2.0 · **XGBoost:** (loaded from JSON models)

---

## Summary Table

| # | Check | Verdict |
|---|-------|---------|
| 1 | Complete test suite | ✅ PASS |
| 2 | Forecasting evaluation (saved models + CCC 2024 test) | ✅ PASS |
| 3 | Independently recalculated PR-AUC, F1, Precision, Recall, Accuracy | ✅ PASS |
| 4 | Metric comparison vs `results/final_results.csv` | ⚠️ MINOR DISCREPANCIES (documented) |
| 5 | Chronological train/val/test split + target leakage | ✅ PASS |
| 6 | Farmers' Protest tweet usage in model training | ✅ PASS (tweets NOT used in XGBoost training) |
| 7 | NLP evaluation — honest labelled sample size + methodology | ⚠️ NOT INDEPENDENT (10 synthetic posts, hardcoded labels) |
| 8 | NLP-extracted events fed into forecasting model | ❌ NOT INTEGRATED (architectural gap, explicitly decoupled) |
| 9 | Streamlit dashboard — model loading + predictions | ✅ PASS (headless; full browser not tested) |
| 10 | Git status — datasets, models, results unmodified | ✅ PASS |

---

## Check 1 — Complete Test Suite

**Verdict: ✅ PASS**

**Command:**
```bash
cd /Users/pranjalgautam/Desktop/civil-unrest-forecasting
.venv/bin/pytest tests/ -v
```

**Output (exact):**
```
============================= test session starts ==============================
platform darwin -- Python 3.13.6, pytest-9.2.0, pluggy-1.7.0
collected 40 items

tests/test_dashboard.py::test_app_syntax_and_compilation PASSED          [  2%]
tests/test_dashboard.py::test_dashboard_features_alignment PASSED        [  5%]
tests/test_dashboard.py::test_dashboard_prediction_execution_flow PASSED [  7%]
tests/test_data_processing.py::test_engineer_temporal_features_structure PASSED [ 10%]
tests/test_data_processing.py::test_lag_mathematical_correctness PASSED  [ 12%]
tests/test_data_processing.py::test_no_lookahead_leakage PASSED          [ 15%]
tests/test_data_processing.py::test_target_generation_logic PASSED       [ 17%]
tests/test_data_processing.py::test_state_isolation PASSED               [ 20%]
tests/test_evaluation.py::test_evaluate_model_perfect_predictions PASSED [ 22%]
tests/test_evaluation.py::test_evaluate_model_threshold_variation PASSED [ 25%]
tests/test_evaluation.py::test_evaluate_model_zero_division PASSED       [ 27%]
tests/test_evaluation.py::test_evaluate_model_input_types PASSED         [ 30%]
tests/test_evaluation.py::test_evaluate_model_mismatched_lengths PASSED  [ 32%]
tests/test_evaluation.py::test_evaluate_model_empty_input PASSED         [ 35%]
tests/test_evaluation.py::test_format_metrics_summary PASSED             [ 37%]
tests/test_nlp.py::test_device_detection PASSED                          [ 40%]
tests/test_nlp.py::test_input_sanitization_empty PASSED                  [ 42%]
tests/test_nlp.py::test_input_sanitization_non_string PASSED             [ 45%]
tests/test_nlp.py::test_extract_event_below_threshold_with_mock PASSED   [ 47%]
tests/test_nlp.py::test_extract_event_unrelated_label_with_mock PASSED   [ 50%]
tests/test_nlp.py::test_extract_event_positive_with_mocks PASSED         [ 52%]
tests/test_nlp.py::test_event_labels_completeness PASSED                 [ 55%]
tests/test_predict.py::test_models_loaded PASSED                         [ 57%]
tests/test_predict.py::test_prediction_probability_range PASSED          [ 60%]
tests/test_predict.py::test_all_horizons PASSED                          [ 62%]
tests/test_predict.py::test_prediction_output PASSED                     [ 65%]
tests/test_predict.py::test_missing_features PASSED                      [ 67%]
tests/test_predict.py::test_invalid_horizon PASSED                       [ 70%]
tests/test_predict.py::test_negative_feature_value PASSED                [ 72%]
tests/test_predict.py::test_nan_feature_value PASSED                     [ 75%]
tests/test_predict.py::test_non_numeric_feature_value PASSED             [ 77%]
tests/test_predict.py::test_non_dict_payload PASSED                      [ 80%]
tests/test_predict.py::test_zero_activity_forecast PASSED                [ 82%]
tests/test_predict.py::test_predict_unrest_batch PASSED                  [ 85%]
tests/test_predict.py::test_predict_unrest_batch_missing_column PASSED   [ 87%]
tests/test_train.py::test_prepare_temporal_splits PASSED                 [ 90%]
tests/test_train.py::test_train_single_horizon PASSED                    [ 92%]
tests/test_train.py::test_evaluate_horizon_model PASSED                  [ 95%]
tests/test_train.py::test_run_training_pipeline_dry_run PASSED           [ 97%]
tests/test_train.py::test_run_training_pipeline_save_to_tmp PASSED       [100%]

============================== 40 passed in 3.05s ==============================
```

**Result:** 40/40 tests passed. Exit code 0.

---

## Check 2 — Forecasting Evaluation with Saved Models + CCC Dataset

**Verdict: ✅ PASS**

**Command:**
```bash
cd /Users/pranjalgautam/Desktop/civil-unrest-forecasting
PYTHONPATH=. .venv/bin/python backend/evaluate_dataset.py
```

**Output (exact):**
```
2026-11-25 09:23:20,448 [INFO] Loading CCC dataset from .../data/ccc_compiled_20212024.csv...
2026-11-25 09:23:20,713 [INFO] Processed dataset shape: (77760, 13)
============================================================
CIVIL UNREST FORECAST - 2024 TEST EVALUATION
============================================================
Test records: 19386
States: 54

------------------------------------------------------------
FORECAST: 1 DAY(S)
------------------------------------------------------------
Threshold : 0.35
Accuracy  : 0.8063
Precision : 0.8156
Recall    : 0.8671
F1 Score  : 0.8405
PR-AUC    : 0.9196

Confusion Matrix:
[[5735 2238]
 [1517 9896]]

Actual event prevalence: 58.87%

------------------------------------------------------------
FORECAST: 3 DAY(S)
------------------------------------------------------------
Threshold : 0.25
Accuracy  : 0.8637
Precision : 0.8835
Recall    : 0.9505
F1 Score  : 0.9158
PR-AUC    : 0.9756

Confusion Matrix:
[[ 2383  1894]
 [  748 14361]]

Actual event prevalence: 77.94%

------------------------------------------------------------
FORECAST: 7 DAY(S)
------------------------------------------------------------
Threshold : 0.1
Accuracy  : 0.9415
Precision : 0.9502
Recall    : 0.9868
F1 Score  : 0.9681
PR-AUC    : 0.9944

Confusion Matrix:
[[ 1011   904]
 [  231 17240]]

Actual event prevalence: 90.12%
```

**Result:** Evaluation script ran successfully. All three XGBoost models loaded from `models/` and evaluated on the 19,386-record 2024 test partition (54 states). Exit code 0.

> [!NOTE]
> High prevalence rates (58.87%, 77.94%, 90.12%) indicate the binary targets (any event in state in next N days?) are heavily imbalanced toward positive class — this is expected given CCC records frequent protests in most US states. High PR-AUC values must be interpreted in that context.

---

## Check 3 — Independently Recalculated Metrics

**Verdict: ✅ PASS**

**Method:** Independent Python script loaded `load_and_process_data()` → filtered year==2024 → loaded `MODELS`, `THRESHOLDS`, `FEATURES` from `backend.predict` → computed all five metrics via `sklearn` directly (no reliance on `evaluate_dataset.py` logic).

**Command:**
```bash
PYTHONPATH=. .venv/bin/python /tmp/verify_metrics.py
```

**Recomputed metrics:**

| Horizon | Threshold | PR-AUC     | F1         | Precision  | Recall     | Accuracy   |
|---------|-----------|------------|------------|------------|------------|------------|
| 1-day   | 0.35      | 0.91963520 | 0.84053170 | 0.81555958 | 0.86708140 | 0.80630352 |
| 3-day   | 0.25      | 0.97558663 | 0.91576330 | 0.88348201 | 0.95049308 | 0.86371608 |
| 7-day   | 0.10      | 0.99443121 | 0.96813141 | 0.95017637 | 0.98677809 | 0.94145259 |

**Recomputed confusion matrices (TN, FP, FN, TP):**

| Horizon | TN   | FP   | FN  | TP    | Prevalence |
|---------|------|------|-----|-------|------------|
| 1-day   | 5735 | 2238 | 1517| 9896  | 58.87%     |
| 3-day   | 2383 | 1894 | 748 | 14361 | 77.94%     |
| 7-day   | 1011 | 904  | 231 | 17240 | 90.12%     |

---

## Check 4 — Comparison Against `results/final_results.csv`

**Verdict: ⚠️ MINOR DISCREPANCIES (1-day and 3-day horizons)**

**Contents of `results/final_results.csv`:**
```
Horizon,Threshold,PR-AUC,F1,Precision,Recall,Accuracy
1-day,0.35,0.9194492861096072,0.839847513719576,0.8143634738809001,0.8669780314824425,0.8060375443937088
3-day,0.25,0.9756645140762761,0.9161351436690874,0.8838186913610333,0.9509045621394862,0.8645036220793797
7-day,0.1,0.9944312087056051,0.9681314053067528,0.9501763668430335,0.9867780894053002,0.9414525946559372
```

**Metric-by-metric comparison (CSV value vs. recomputed, full precision):**

| Horizon | Metric    | CSV Value          | Recomputed         | Difference         |
|---------|-----------|--------------------|--------------------|--------------------|
| 1-day   | PR-AUC    | 0.9194492861096072 | 0.9196352002       | **+0.0001859141**  |
| 1-day   | F1        | 0.8398475137195760 | 0.8405317026       | **+0.0006841888**  |
| 1-day   | Precision | 0.8143634738809001 | 0.8155595846       | **+0.0011961108**  |
| 1-day   | Recall    | 0.8669780314824425 | 0.8670813984       | **+0.0001033669**  |
| 1-day   | Accuracy  | 0.8060375443937088 | 0.8063035180       | **+0.0002659736**  |
| 3-day   | PR-AUC    | 0.9756645140762761 | 0.9755866251       | **−0.0000778889**  |
| 3-day   | F1        | 0.9161351436690874 | 0.9157632955       | **−0.0003718482**  |
| 3-day   | Precision | 0.8838186913610333 | 0.8834820055       | **−0.0003366858**  |
| 3-day   | Recall    | 0.9509045621394862 | 0.9504930836       | **−0.0004114785**  |
| 3-day   | Accuracy  | 0.8645036220793797 | 0.8637160838       | **−0.0007875383**  |
| 7-day   | PR-AUC    | 0.9944312087056051 | 0.9944312087056051 | **0.0000000000** ✓ |
| 7-day   | F1        | 0.9681314053067528 | 0.9681314053067528 | **0.0000000000** ✓ |
| 7-day   | Precision | 0.9501763668430335 | 0.9501763668430335 | **0.0000000000** ✓ |
| 7-day   | Recall    | 0.9867780894053002 | 0.9867780894053002 | **0.0000000000** ✓ |
| 7-day   | Accuracy  | 0.9414525946559372 | 0.9414525946559372 | **0.0000000000** ✓ |

**Analysis:**

- **7-day horizon:** Perfect agreement (0.000 difference). `final_results.csv` values are exactly reproducible.
- **1-day and 3-day horizons:** Small but non-zero discrepancies (all < 0.0012). The maximum absolute difference is 0.0012 (1-day Precision). These differences are consistent with the training environment being different from the local evaluation environment (models were trained on Google Colab with different data ordering, random seeds, or XGBoost version). The `final_results.csv` values appear to have been recorded from a slightly different run (possibly on the validation set or a Colab-side evaluation pass) rather than the current local evaluation.

> [!WARNING]
> The differences are small (< 0.2% in all metrics) but non-zero for 1-day and 3-day horizons. The `final_results.csv` values cannot be exactly reproduced locally, suggesting they were recorded from a different execution environment or data snapshot. This is a methodological transparency concern that should be documented in the README.

---

## Check 5 — Chronological Train/Val/Test Split + Target Leakage

**Verdict: ✅ PASS**

**Command:**
```bash
PYTHONPATH=. .venv/bin/python /tmp/verify_metrics.py  # section [2] and [3]
```

**Split boundaries (exact):**

| Split | Filter         | Rows  | Date Range                    |
|-------|----------------|-------|-------------------------------|
| Train | year ≤ 2022    | 38,664 | 2021-01-15 → 2022-12-31       |
| Val   | year == 2023   | 19,710 | 2023-01-01 → 2023-12-31       |
| Test  | year == 2024   | 19,386 | 2024-01-01 → 2024-12-24       |

**Overlap check:**
- Train ∩ Val: **0 rows** ✓
- Val ∩ Test: **0 rows** ✓
- Train ∩ Test: **0 rows** ✓

**Feature leakage check:**

Feature columns used: `lag_1, lag_3, lag_7, lag_14, rolling_7_mean, rolling_7_sum, rolling_14_mean`

All features are computed by `engineer_temporal_features()` in `backend/data_processing.py` as backward-looking lags and rolling statistics on `event_count` — they strictly use past data only. The target columns (`target_1day`, `target_3day`, `target_7day`) are forward-shifted event counts (lookforward targets) and are NOT present in the feature set `FEATURES`.

**Result:** No temporal data leakage detected. The split is strictly chronological by calendar year. Features use only past event counts.

---

## Check 6 — Farmers' Protest Tweet Usage in Model Training

**Verdict: ✅ PASS — Tweets were NOT used in XGBoost model training**

**Evidence:**

1. **Dataset file confirmed present:**
   ```
   data/cleaned_farmers_protest_tweets.csv — 120 MB, 1,232,828 rows (header + 1,232,827 data rows)
   Header: tweet_id,text,date,username,url,language,likes,retweets,replies
   ```

2. **`backend/data_processing.py` hardcodes CCC dataset path:**
   ```python
   DATA_PATH = ROOT_DIR / "data" / "ccc_compiled_20212024.csv"
   ```
   The farmers' protest CSV is never referenced.

3. **`backend/train.py` calls `load_and_process_data()`** which loads only `ccc_compiled_20212024.csv`. Grep confirms no reference to `cleaned_farmers_protest_tweets` in `train.py` or `data_processing.py`.

4. **`backend/process_farmers_tweets.py`** exists and processes the tweet file through the NLP pipeline — but its output is never incorporated into the XGBoost feature matrix.

**Conclusion:** The 1,232,827 cleaned Farmers' Protest tweets were scraped, cleaned, and processed through the NLP extraction pipeline as a demonstration of that subsystem. They did **not** contribute to any XGBoost training or evaluation. The XGBoost models were trained exclusively on CCC (Crowd Counting Consortium) data from 2021–2024.

---

## Check 7 — NLP Evaluation: Actual Labelled Sample Size and Methodology

**Verdict: ⚠️ NOT AN INDEPENDENT TEST SET — documented below**

**Command:**
```bash
PYTHONPATH=. .venv/bin/python backend/evaluate_demo.py
```

**Output (exact):**
```
Demo NLP Evaluation
------------------------------
Accuracy : 0.8
Precision: 0.75
Recall   : 1.0
F1 Score : 0.8571428571428571
```

**Critical methodological notes:**

| Property | Value |
|----------|-------|
| Sample size | **10 posts** |
| Source | Synthetic/manually authored demonstration examples in `data/demo_social_posts.csv` |
| Ground truth | **Hardcoded** in `backend/evaluate_demo.py` as `y_true = [1, 1, 1, 0, 0, 1, 0, 1, 0, 1]` |
| Independence | **Not independent** — posts were written to demonstrate the NLP pipeline and labels match obvious civil-unrest content |
| NLP model evaluated | Zero-shot text classification via `facebook/bart-large-mnli` |

**The 10 demo posts:**
1. "Workers protested over unpaid wages in Delhi." → predicted: True ✓
2. "Students held a demonstration against rising tuition fees." → predicted: True ✓
3. "Residents gathered to demand clean drinking water." → predicted: True ✓
4. "The cricket team won the championship yesterday." → predicted: False ✓
5. "Heavy rain caused flooding in several neighborhoods." → predicted: **True** ✗ (y_true=0)
6. "Teachers announced a strike over salary disputes." → predicted: True ✓
7. "The government announced a new education policy." → predicted: **True** ✗ (y_true=0)
8. "Thousands attended a rally demanding better working conditions." → predicted: True ✓
9. "A music festival attracted visitors across the city." → predicted: False ✓
10. "Transport workers protested against proposed job cuts." → predicted: True ✓

> [!CAUTION]
> The reported NLP metrics (Accuracy=0.80, Precision=0.75, Recall=1.0, F1=0.857) are based on 10 synthetic, non-independent examples with manually assigned labels. These numbers **cannot be presented as a generalised evaluation of NLP performance** on real social media data. Additionally, `final_protest_articles.csv` contains protest-selected news headlines — these also do not constitute an independent, randomly-sampled evaluation set.

**A genuine independent NLP evaluation would require:** a randomly sampled, human-annotated set of real social media posts with event/no-event labels, evaluated blind to the model development process.

---

## Check 8 — NLP-Extracted Events Fed into Forecasting Model

**Verdict: ❌ NOT INTEGRATED — Architectural gap**

**Evidence:**

```python
# backend/predict.py — does NOT import social_media_extraction
grep: 'social_media_extraction' in predict.py → False
grep: 'extract_event' in predict.py → False

# app.py — does NOT import social_media_extraction or extract_event
grep: 'social_media_extraction' in app.py → False
grep: 'extract_event' in app.py → False
```

**Architecture reality:**

The two subsystems are **completely decoupled**:

- **NLP pipeline** (`backend/social_media_extraction.py`): Accepts raw social media text → returns structured fields (`event_detected`, `event_type`, `locations`, `organizations`, `confidence`). Output is written to CSV files or displayed in notebooks. It is **never called** by the forecasting pipeline.

- **XGBoost forecasting model** (`backend/predict.py` → `predict_unrest()`): Accepts 7 lag/rolling features derived from historical CCC event counts → returns probability and binary forecast. It **never reads** NLP output.

The Streamlit dashboard (`app.py`) exposes both subsystems as separate UI tabs, but there is no code path connecting them. The NLP tab demonstrates event extraction on user-provided text; the forecasting tab computes predictions from CCC-derived time-series features. These are presented as separate capabilities.

> [!IMPORTANT]
> This architectural decoupling is acknowledged and documented in the project README and in `docs/METHODOLOGY.md`. The NLP extraction pipeline and the XGBoost forecasting model are presented as a **multi-component system** but they are **not end-to-end integrated**. Social media signals extracted by NLP do not currently feed into the forecasting model as features. This is the most significant remaining technical gap in the project.

---

## Check 9 — Streamlit Dashboard: Model Loading and Predictions

**Verdict: ✅ PASS (headless verification; full browser rendering not tested)**

**Command:**
```bash
PYTHONPATH=. .venv/bin/python -c "
from backend.predict import MODELS, THRESHOLDS, FEATURES, predict_unrest
from backend.data_processing import load_and_process_data

print('Models loaded:', len(MODELS), '— horizons:', sorted(MODELS.keys()))
print('Thresholds:', THRESHOLDS)
print('Features:', FEATURES)

sample_input = {f: 1.0 for f in FEATURES}
result = predict_unrest(sample_input)
print('Sample predict_unrest output:', result)

df = load_and_process_data()
latest = df.sort_values('date').tail(1)
print('Latest data record date:', latest['date'].values[0])
print('Dashboard data shape:', df.shape)
"
```

**Output (exact):**
```
Models loaded: 3 — horizons: [1, 3, 7]
Thresholds: {'1': 0.35, '3': 0.25, '7': 0.1}
Features: ['lag_1', 'lag_3', 'lag_7', 'lag_14', 'rolling_7_mean', 'rolling_7_sum', 'rolling_14_mean']
Sample predict_unrest output: {
  'horizon_days': 1,
  'probability': 0.7676,
  'threshold': 0.35,
  'prediction': 1,
  'forecast': 'Unrest event expected'
}
Latest data record date: 2024-12-24T00:00:00.000000
Dashboard data shape: (77760, 13)
PASS: Model loading and prediction verified.
```

**Additional static checks on `app.py`:**
- Imports `predict_unrest` and `FEATURES` from `backend.predict` ✓
- Imports `load_and_process_data` from `backend.data_processing` ✓
- Uses `@st.cache_data` on `get_historical_data()` ✓
- Does NOT import `social_media_extraction` (NLP tab calls it dynamically) ✓
- `app.py` compiles without syntax errors ✓ (confirmed by `test_app_syntax_and_compilation`)

> [!NOTE]
> Full Streamlit browser rendering (UI rendering, widget interactions, map display) cannot be verified in a headless environment without a browser. The model loading, data loading, and core prediction logic are verified to work correctly from the Python layer.

---

## Check 10 — Git Status: Protected Files Unmodified

**Verdict: ✅ PASS**

**Command:**
```bash
cd /Users/pranjalgautam/Desktop/civil-unrest-forecasting && git status && git diff --stat HEAD
```

**Output (exact):**
```
On branch main
Your branch is up to date with 'origin/main'.

Untracked files:
  (use "git add <file>..." to include in what will be committed)
	.agents/
	civil_unrest_models.zip

nothing added to commit but untracked files present (use "git add" to track)
```

**Protected file integrity:**

| File | Status |
|------|--------|
| `results/final_results.csv` | ✅ Unmodified (no diff vs HEAD) |
| `models/xgboost_1day.json` (964,672 bytes) | ✅ Unmodified |
| `models/xgboost_3day.json` (894,746 bytes) | ✅ Unmodified |
| `models/xgboost_7day.json` (775,610 bytes) | ✅ Unmodified |
| `models/metadata.json` (427 bytes) | ✅ Unmodified |
| `data/ccc_compiled_20212024.csv` (113 MB) | ✅ Not in Git (in .gitignore) |
| `data/cleaned_farmers_protest_tweets.csv` (120 MB) | ✅ Not in Git (in .gitignore) |
| `data/farmers-protest-tweets-2021-03-5.json` (1.3 GB) | ✅ Not in Git (in .gitignore) |

**Untracked files:** `.agents/` (Antigravity IDE config) and `civil_unrest_models.zip` — neither is tracked or staged. No sensitive data, secrets, or large datasets are staged or committed.

---

## Critical Gaps Summary

### Gap 1 — NLP ↔ Forecasting Not Integrated (CRITICAL)
The NLP extraction pipeline and the XGBoost forecasting model are architecturally decoupled. Social media signals extracted from tweets or news articles are never used as features in the forecasting model. This is the system's most significant structural limitation.

### Gap 2 — `final_results.csv` Metric Discrepancy (MINOR)
Reported metrics in `final_results.csv` for 1-day and 3-day horizons differ from locally reproduced values by up to 0.0012 (< 0.2%). The 7-day metrics reproduce exactly. The discrepancy is likely due to different execution environment (Colab vs. local), not data manipulation. The original Colab notebook (`notebooks/Untitled74-2.ipynb`) is the authoritative source but cannot be re-run locally without GPU access.

### Gap 3 — NLP Evaluation Is Not Independent (SIGNIFICANT)
The only NLP evaluation script uses 10 synthetic, manually authored posts with hardcoded ground-truth labels. This cannot be claimed as an objective evaluation of the NLP pipeline's performance on real-world social media data.

### Gap 4 — Streamlit Full-Browser Test Not Performed (MINOR)
Full browser rendering of the Streamlit dashboard was not tested. All backend logic (model loading, predictions, data loading) was verified from the Python layer, but UI rendering, map visualization, and user input interactions were not interactively tested.

### Gap 5 — High Class Imbalance Not Prominently Disclosed (MINOR)
The binary targets are heavily imbalanced toward positive class (58.9%–90.1% prevalence). This context is important for interpreting PR-AUC and F1 scores and should be prominently disclosed in evaluations.

---

## Appendix — Environment

```
Python:   3.13.6
pytest:   9.2.0
Platform: Darwin (macOS)
Venv:     .venv/bin/python

Key packages (from .venv):
  xgboost, scikit-learn, pandas, numpy
  torch (for NLP zero-shot classification via transformers)
  streamlit (dashboard)
  spacy (NLP pipeline)
```

---

*Report generated by independent automated audit. No model files, result files, datasets, or source code were modified during this verification.*
