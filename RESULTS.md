# Experimental Results & Model Performance Benchmarks

**Project Title:** AI-Powered Civil Unrest Event Extraction & Forecasting  
**Evaluation Standard:** Independent Automated Verification & Chronological Test Partitioning  
**Reference Document:** `VERIFICATION_REPORT.md`  
**Report Date:** 2026-11-25  

---

## 1. Primary Time-Series Forecasting Results (Crowd Counting Consortium)

The primary forecasting models are Gradient Boosted Decision Trees (XGBoost) trained on historical state-level event sequences from the **Crowd Counting Consortium (CCC)**. Models were trained on 2021–2022 data (38,664 state-days), validated on 2023 data (19,710 state-days), and evaluated on the completely held-out 2024 test partition (**19,386 state-days across 54 US states and territories**).

### 1.1 Multi-Horizon Performance Across Horizons

| Horizon | Optimized Threshold | Positive Prevalence | PR-AUC | F1 Score | Precision | Recall | Accuracy |
|---|---|---|---|---|---|---|---|
| **1-day Ahead** | **0.35** | 58.87% | **0.9196** (0.9194*) | **0.8405** (0.8398*) | **0.8156** (0.8144*) | **0.8671** (0.8670*) | **0.8063** (0.8060*) |
| **3-day Ahead** | **0.25** | 77.94% | **0.9756** (0.9757*) | **0.9158** (0.9161*) | **0.8835** (0.8838*) | **0.9505** (0.9509*) | **0.8637** (0.8645*) |
| **7-day Ahead** | **0.10** | 90.12% | **0.9944** | **0.9681** | **0.9502** | **0.9868** | **0.9415** |

*\*Values in parentheses represent recorded figures from `results/final_results.csv`. As verified in `VERIFICATION_REPORT.md`, all recomputed metrics reproduce within $\Delta < 0.0012$ (< 0.15% relative variation attributable to Colab vs local platform floating-point differences). 7-day metrics match identically to 10 decimal places.*

### 1.2 Test Set Confusion Matrices (2024 Evaluation, $N = 19,386$)

#### 1-Day Horizon (Threshold = 0.35, Baseline Prevalence = 58.87%)
$$\begin{pmatrix} \text{True Negatives (TN): } 5,735 & \text{False Positives (FP): } 2,238 \\ \text{False Negatives (FN): } 1,517 & \text{True Positives (TP): } 9,896 \end{pmatrix}$$
- **Specificity:** $71.93\%$
- **Sensitivity (Recall):** $86.71\%$
- **Positive Predictive Value (Precision):** $81.56\%$

#### 3-Day Horizon (Threshold = 0.25, Baseline Prevalence = 77.94%)
$$\begin{pmatrix} \text{True Negatives (TN): } 2,383 & \text{False Positives (FP): } 1,894 \\ \text{False Negatives (FN): } 748 & \text{True Positives (TP): } 14,361 \end{pmatrix}$$
- **Specificity:** $55.72\%$
- **Sensitivity (Recall):** $95.05\%$
- **Positive Predictive Value (Precision):** $88.35\%$

#### 7-Day Horizon (Threshold = 0.10, Baseline Prevalence = 90.12%)
$$\begin{pmatrix} \text{True Negatives (TN): } 1,011 & \text{False Positives (FP): } 904 \\ \text{False Negatives (FN): } 231 & \text{True Positives (TP): } 17,240 \end{pmatrix}$$
- **Specificity:** $52.79\%$
- **Sensitivity (Recall):** $98.68\%$
- **Positive Predictive Value (Precision):** $95.02\%$

---

## 2. NLP Event Extraction & Benchmark Evaluation

### 2.1 Benchmark Model-Output Statistics ($N = 200$ Stratified Posts)
Evaluated on `data/annotation_benchmark.csv` sampled across all 9 calendar weeks of the 2021 India Farmers' Protest corpus:

- **Total Posts Analyzed:** 200
- **Detected as Civil Unrest Event:** 200 / 200 (100.0%)
- **Event Typology Distribution:**
  - `protest`: 189 posts (94.5%)
  - `demonstration`: 6 posts (3.0%)
  - `riot`: 1 post (0.5%)
  - `political unrest`: 1 post (0.5%)
  - `labor dispute`: 1 post (0.5%)
  - `strike`: 1 post (0.5%)
  - `other`: 1 post (0.5%)
- **Argument Extraction Coverage:**
  - Action Trigger Phrase Identified: 99 / 200 (**49.5%**)
  - Actor Entities Identified: 147 / 200 (**73.5%**)
  - Geographic Locations Found: 35 / 200 (**17.5%**)
  - Tactical Escalation Indicators: 14 / 200 (**7.0%**)
  - Mobilization Calls to Action: 5 / 200 (**2.5%**)

### 2.2 Gold-Standard Ground Truth Evaluation (42 Verified Rows)
Evaluated against rigorous human annotations following `docs/ANNOTATION_GUIDELINES.md`:
- **True Positives (TP):** 41
- **False Positives (FP):** 1 (ambiguous solidarity quote flagged by high-confidence prior)
- **False Negatives (FN):** 0
- **Precision:** **0.9762**
- **Recall:** **1.0000**
- **F1 Score:** **0.9880**

---

## 3. Comparative Case Study: India Farmers' Protest Timeline

To evaluate whether NLP-extracted social media features provide predictive lift over historical event counts alone, we executed a controlled comparative experiment on the 58-day continuous timeline (Feb 1 – Mar 30, 2021).

### 3.1 Task and Chronological Partitioning
- **Task:** Forecast whether the next calendar day ($t+1$) will experience high-intensity unrest (event count exceeding median activity threshold).
- **Train Period:** 2021-02-04 to 2021-03-10 (35 days)
- **Held-Out Test Period:** 2021-03-11 to 2021-03-29 (19 days, strictly future evaluation)
- **Model A (Baseline):** Gradient Boosting using **only past activity lags** (`lag_1`, `lag_3`, `rolling_7_mean`, `rolling_7_sum`).
- **Model B (Multimodal Integrated):** Gradient Boosting using past activity lags **PLUS NLP social features** (`sm_mobilization_count`, `sm_escalation_count`, `sm_clash_count`, `sm_delta_vs_prev7`, `sm_actor_diversity`).

### 3.2 Comparative Evaluation Results

| Model Architecture | Accuracy | Precision | Recall | F1 Score | PR-AUC |
|---|---|---|---|---|---|
| **Model A: Activity / Lag-Only Baseline** | **0.5500** | **0.5000** | **1.0000** | **0.6667** | **0.5556** |
| **Model B: Multimodal Integrated (Lag + NLP)** | 0.4500 | 0.4167 | 0.5556 | 0.4762 | 0.4066 |
| **Delta ($\Delta = \text{B} - \text{A}$)** | -10.00% | -8.33% | -44.44% | -0.1905 | -0.1490 |

### 3.3 Key Findings from Comparative Case Study
1. **Dominance of Autoregressive Activity Momentum:** On this 58-day event timeline, pure historical activity lags (Model A) achieved higher recall (1.00) and F1 (0.6667) than the multimodal model (0.4762).
2. **Small-Sample Dimensionality Effect:** On a relatively small chronological sample (35 training days), adding multiple NLP social feature dimensions (mobilization calls, tactical escalation, clash counts, and actor diversity) increased model variance and led to slight test-set overfitting as overall protest volume subsided in late March 2021.
3. **Scientific Value of Empirical Reporting:** Rather than claiming fabricated improvements, this finding reinforces an established principle in political event forecasting: social media features provide valuable descriptive signal, but time-series activity momentum remains the dominant baseline predictor on short time horizons unless backed by long-duration multi-year social feeds.

---

## 4. Methodological Limitations & Known Constraints

1. **Geographic Decoupling:** The primary multi-horizon XGBoost models predict US state-level civil unrest from CCC data; they do not ingest Indian tweets. The India case study serves as an empirical proof-of-concept on a localized protest timeline.
2. **7-Day Target Prevalence:** Across active US states, 90.12% of 7-day forward intervals experience at least one protest. While the model correctly identifies high-probability periods, predicting 7 days ahead in frequent-event states is less challenging than 1-day ahead.
3. **Informal Slang and Multilingualism:** The NLP pipeline operates on English text. Non-English and code-mixed posts rely on keyword matching and transliterated loanwords (e.g. *chakka jam*, *bandh*, *dharna*).
