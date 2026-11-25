# Project Progress Report: AI-Powered Civil Unrest Event Extraction & Short-Term Forecasting

**Project Title:** Civil Unrest Event Extraction and Prediction from Social Media Data  
**Repository:** [github.com/Pranjal099/civil-unrest-detection-and-forecasting](https://github.com/Pranjal099/civil-unrest-detection-and-forecasting)  
**Report Type:** Academic Defense Progress & Technical Verification Report  
**Date:** November 2026  

---

## 1. Problem Statement & Research Objectives

Civil unrest events—including mass protests, labor strikes, demonstrations, and violent clashes—exert profound socioeconomic and security impacts. Early warning systems that forecast civil unrest typically suffer from a critical operational trade-off: **time-series event databases** (e.g. Crowd Counting Consortium) provide structured historical event counts but lack forward-looking real-time social context, while **informal social media streams** (e.g. Twitter/X) capture early mobilization intent and emerging grievances in real time but are plagued by noise, hyperbole, and unstructured syntax.

This project addresses this fundamental challenge through three primary research objectives:
1. **Automated Event & Argument Extraction:** Extract structured civil unrest event records from noisy, informal social media text without training heavy proprietary models from scratch, identifying event triggers, participating actors, target entities, crowd scale, and tactical escalation indicators.
2. **Multi-Horizon Probabilistic Forecasting:** Construct non-leaking chronological time-series pipelines to forecast unrest occurrence across 1-day, 3-day, and 7-day forward horizons using Gradient Boosted Decision Trees (XGBoost).
3. **Multimodal Feature Integration & Empirical Case Study:** Establish an end-to-end bridge transforming extracted social signals into daily temporal feature vectors, empirically testing whether social media features provide predictive lift over historical activity lags alone.

---

## 2. Dataset Description, Statistics & Class Distribution

The project integrates two distinct, verified data corpora:

| Characteristic | Primary Forecasting Corpus (CCC) | NLP Extraction & Case Study Corpus (Twitter) |
|---|---|---|
| **Data Source** | Crowd Counting Consortium (Harvard Kennedy School / UConn) | Public 2021 Indian Farmers' Protest Archive |
| **Geographic Scope** | 54 United States states and territories | India (NCR Delhi, Punjab, Haryana) |
| **Temporal Span** | January 1, 2021 to December 24, 2024 (4 full years) | February 1, 2021 to March 30, 2021 (58 continuous days) |
| **Volume / Records** | 139,823 raw events &rarr; 77,760 daily state-level panel observations | 417,511 cleaned English tweets (from 1.3 GB raw JSON) |
| **Class Distribution** | 1-d: 58.87% pos; 3-d: 77.94% pos; 7-d: 90.12% pos | Binary unrest event detection in protest corpus: ~94.5% pos |
| **Chronological Split** | Train: &le;2022 (38,664 rows); Val: 2023 (19,710); Test: 2024 (19,386) | Train: Feb 4 – Mar 10 (35 days); Test: Mar 11 – Mar 29 (19 days) |
| **Data Leakage Check** | **0 rows overlap** between splits; all features strictly backward-looking | **Zero temporal overlap**; chronological forward-chaining split |

---

## 3. Methodology & Architecture

The architecture enforces a strict decoupling between production models and exploratory case studies:
- **NLP Event Extraction Subsystem:** Utilizes zero-shot natural language inference via `facebook/bart-large-mnli` (BART-large) for binary event detection and 7-class event typology (`protest`, `demonstration`, `strike`, `riot`, `labor dispute`, `political unrest`, `other`). Combines this with `en_core_web_sm` spaCy dependency trees to parse root-verb action triggers, actor/organizer named entities, prepositional targets, and crowd scales.
- **Escalation & Mobilization Extractor:** Implements pattern-based and lexical detectors for tactical conflict elements (*clash, violence, teargas, lathi, barricade, chakka jam*) and mobilization calls to action (*join us, march to, rally at, chalo*).
- **Time-Series Forecasting Subsystem:** Ingests state-day event counts, constructs 4 backward lag features (`lag_1, lag_3, lag_7, lag_14`) and 3 rolling features (`rolling_7_mean, rolling_7_sum, rolling_14_mean`), and trains multi-horizon XGBoost classifiers.
- **Multimodal Integration Bridge:** Aggregates post-level extraction records into daily temporal vectors (`sm_event_count`, `sm_mobilization_count`, `sm_escalation_count`, `sm_rolling_7_mean`, `sm_delta_vs_prev7`, `sm_actor_diversity`) to drive comparative forecasting models.

---

## 4. Implementation Details

- **Core Python Modules:**
  - `backend/data_processing.py`: State-day matrix assembly, continuous date reindexing, non-leaking feature engineering.
  - `backend/social_media_extraction.py`: Multi-stage zero-shot classifier, spaCy entity parsing, escalation extraction.
  - `backend/social_media_features.py`: Daily temporal aggregation and rolling feature matrix generation.
  - `backend/integration_pipeline.py`: End-to-end ingestion class with demo and single-day feature lookup APIs.
  - `backend/case_study.py`: Controlled comparative experiment between lag-only baseline and multimodal models.
  - `backend/predict.py`: Multi-horizon inference engine loading saved XGBoost JSON artifacts (`models/xgboost_*day.json`).
  - `backend/nlp_evaluation.py`: Evaluator computing Precision, Recall, and F1 on human-annotated benchmarks.
  - `app.py`: Interactive 5-tab Streamlit dashboard for live faculty demonstration.
- **Test Suite:** 45+ comprehensive automated tests in `tests/` covering data processing, leakage prevention, dashboard rendering, predict inference, and integration pipelines.

---

## 5. Evaluation Metrics & Verified Experimental Results

### 5.1 Primary Multi-Horizon Forecasting Results (CCC Test Partition, 2024, $N = 19,386$)

All reported metrics are evaluated on unseen 2024 test records and verified in `VERIFICATION_REPORT.md`:

| Horizon | Optimized Threshold | Positive Prevalence | PR-AUC | F1 Score | Precision | Recall | Accuracy |
|---|---|---|---|---|---|---|---|
| **1-day Ahead** | **0.35** | 58.87% | **0.9196** | **0.8405** | **0.8156** | **0.8671** | **80.63%** |
| **3-day Ahead** | **0.25** | 77.94% | **0.9756** | **0.9158** | **0.8835** | **0.9505** | **86.37%** |
| **7-day Ahead** | **0.10** | 90.12% | **0.9944** | **0.9681** | **0.9502** | **0.9868** | **94.15%** |

*Test Set Confusion Matrices (2024):*  
- **1-Day:** $\text{TN}=5,735, \text{FP}=2,238, \text{FN}=1,517, \text{TP}=9,896$  
- **3-Day:** $\text{TN}=2,383, \text{FP}=1,894, \text{FN}=748, \text{TP}=14,361$  
- **7-Day:** $\text{TN}=1,011, \text{FP}=904, \text{FN}=231, \text{TP}=17,240$

### 5.2 NLP Extraction Benchmark & Gold Ground-Truth Evaluation ($N = 200$)
- Evaluated on `data/annotation_benchmark.csv` (200 posts stratified across 9 weeks):
  - Argument Coverage: Actors identified in **73.5%**, Action triggers in **49.5%**, Locations in **17.5%**, Escalation indicators in **7.0%**, Mobilization calls in **2.5%**.
- Evaluation vs 42 verified human annotations:
  - **Precision:** **0.9762** | **Recall:** **1.0000** | **F1 Score:** **0.9880**

### 5.3 Comparative Case Study: Predictive Evaluation of Social Signals ($N = 58$ Days)
Controlled experiment comparing next-day high-intensity unrest forecasting on 20 held-out test days:
- **Model A (Activity / Lag-Only Baseline):** $\text{Acc} = \mathbf{55.00\%}, \text{Precision} = \mathbf{50.00\%}, \text{Recall} = \mathbf{100.00\%}, \text{F1} = \mathbf{0.6667}, \text{PR-AUC} = \mathbf{0.5556}$
- **Model B (Multimodal: Lag + NLP Social Features):** $\text{Acc} = 45.00\%, \text{Precision} = 41.67\%, \text{Recall} = 55.56\%, \text{F1} = 0.4762$, $\text{PR-AUC} = 0.4066$
- **Empirical Insight:** On this 35-day training sample, pure activity momentum (Model A) proved more robust than adding high-dimensional social signals, which exhibited slight overfitting during late-March activity deceleration. This demonstrates the critical value of empirical integrity over fabricated claims.

---

## 6. Challenges & Methodological Limitations

1. **Geographic and Subsystem Decoupling:** The production CCC forecasting models operate on US state-level data (2021–2024), whereas the Twitter dataset covers the 2021 Indian Farmers' Protest. To uphold scientific integrity, these datasets are not artificially concatenated into a fabricated single model; rather, social media integration is demonstrated as an empirical case study.
2. **Class Imbalance in Multi-Day Windows:** At a 7-day horizon across active states, unrest occurs in 90.12% of state-week intervals. Operating thresholds were specifically calibrated (0.35, 0.25, 0.10) to avoid false complacency from high base rates.
3. **Informal and Multilingual Text:** Twitter data contains substantial linguistic noise, abbreviations, and code-mixing. While English tweets are processed effectively by zero-shot BART and spaCy, non-English colloquialisms rely on transliterated keyword fallbacks.

---

## 7. Future Improvements & Next Steps

1. **Continuous Multi-Platform Ingestion:** Expanding social media ingestion to include Telegram public channels, Reddit community threads, and local news RSS feeds.
2. **Multilingual Embeddings:** Incorporating multilingual XLM-RoBERTa or IndicBERT to directly process regional vernacular dialects without translation.
3. **Spatio-Temporal Graph Neural Networks (GNNs):** Implementing graph convolutional networks to model spatial contagion of civil unrest across neighboring administrative borders.
