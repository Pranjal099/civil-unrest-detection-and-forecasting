# Project Audit & Requirements Mapping

**Project Title:** Civil Unrest Event Extraction and Prediction from Social Media Data  
**Repository:** `civil-unrest-detection-and-forecasting`  
**Audit Date:** 2026-11-25  
**Evaluation Target:** Faculty Demonstration & Academic Defense Readiness  

---

## 1. Executive Summary

This audit establishes a rigorous requirement-to-implementation trace matrix for the civil unrest forecasting system. The project encompasses two major subsystems:
1. **NLP Event Extraction Subsystem:** Extracts structured event records, actor entities, trigger phrases, targets, crowd scale, and escalation indicators from noisy, informal social media posts.
2. **Temporal Forecasting Subsystem:** Produces multi-horizon (1-day, 3-day, 7-day) probabilistic forecasts of civil unrest occurrence across geographic regions using temporal sequence modeling.

To maintain scientific integrity and academic transparency, this audit explicitly documents what is **COMPLETE**, what is **PARTIAL**, what was **PRESERVED**, and what **LIMITATIONS** exist regarding dataset provenance and cross-system integration.

---

## 2. Requirement-to-Implementation Checklist

| Requirement | Scope Description | Status | Implementation Details & Code References | Key Limitations & Scope Boundary |
|---|---|---|---|---|
| **A. Civil Unrest Event Extraction** | Detect civil unrest signals from informal social media text | **COMPLETE** | Implemented via zero-shot classification using `facebook/bart-large-mnli` coupled with defensive sanitization and keyword anchoring in `backend/social_media_extraction.py` (`extract_event()`). | Evaluated on English social media posts. Informal slang and multilingual code-mixing (e.g. Hinglish) rely on lexical fallbacks. |
| **B. Event Triggers & Typology** | Identify event triggers and categorize into protests, demonstrations, marches, clashes, strikes | **COMPLETE** | Event typology classification uses 7 canonical categories (`protest`, `demonstration`, `strike`, `riot`, `labor dispute`, `political unrest`, `other`). Action triggers are extracted using root-verb dependency parsing and keyword matching in `_extract_spacy_fields()`. | Minor typology overlaps (e.g., protest vs demonstration) in zero-shot probability distributions. |
| **C. Argument Extraction** | Extract Actor/organizer, Location, Date/time, Target, Scale | **COMPLETE** | Multi-entity extraction pipeline via spaCy `en_core_web_sm` and targeted dependency parsing: `locations` (GPE/LOC), `organizations`/`actors` (ORG/PERSON), `dates` (DATE/TIME), `target`/`grievance` (prepositional objects), and `scale` (CARDINAL crowd quantifiers). | Scale is only extractable when explicit numeric or crowd adjectives (e.g., "thousands", "50,000") appear in text. |
| **D. Escalation & Mobilization** | Identify mobilization calls, grievances, clash reports, and escalating indicators | **COMPLETE** | Pattern-based and lexical escalation detector in `backend/social_media_extraction.py` tracking tactical keywords (`clash`, `violence`, `teargas`, `lathi`, `arrest`, `barricade`, `chakka jam`) and mobilization solicitations (`join us`, `march to`, `chalo`, `rally at`). | Captures explicit textual signals; covert coordination or indirect sarcasm is outside lexical scope. |
| **E. Temporal Event Sequences** | Aggregate structured events into temporal sequence matrices | **COMPLETE** | `backend/data_processing.py` constructs chronological lag (1, 3, 7, 14 days) and rolling window (7-day, 14-day) sequences for CCC state-level data. `backend/social_media_features.py` builds daily aggregated timelines for social posts. | Social media temporal sequences are continuous only within the 58-day window of available Twitter data. |
| **F. Forecasting Unrest Emergence** | Forecast emergence or escalation across 1-day, 3-day, and 7-day horizons | **COMPLETE** | Multi-horizon Gradient Boosted Decision Trees (XGBoost) trained on CCC state-level historical sequences (`models/xgboost_*day.json`). Threshold-optimized operating points: 0.35 (1-day), 0.25 (3-day), 0.10 (7-day). Comparative case study in `backend/case_study.py`. | Production models forecast US state-level events from CCC data. Integrated social-media forecasting is demonstrated as an empirical case study on the Farmers' Protest timeline. |
| **G. Evaluation & Metrics** | Comprehensive evaluation using PR-AUC, F1, Precision, Recall, Accuracy | **COMPLETE** | Forecasting evaluated on held-out 2024 test partition (19,386 records). Independently verified in `VERIFICATION_REPORT.md`. NLP evaluated via 200-item benchmark (`data/annotation_benchmark.csv`) and gold-verified sample in `backend/nlp_evaluation.py`. | Initial synthetic 10-post evaluation replaced by genuine 200-post stratified benchmark with separate human ground-truth schema. |

---

## 3. Subsystem Architecture & Dataset Provenance

### 3.1 Dataset Provenance & Geopolitical Reality
1. **Crowd Counting Consortium (CCC) Corpus (`data/ccc_compiled_20212024.csv`):**
   - **Coverage:** 139,823 verified civil unrest events across 54 United States states and territories from January 2021 through December 2024.
   - **Role:** Ground truth for training, validating, and testing the primary multi-horizon XGBoost forecasting models.
   - **Split:** Strictly chronological (Train: 2021–2022, Validation: 2023, Test: 2024). Zero temporal overlap.

2. **India Farmers' Protest Twitter Corpus (`data/cleaned_farmers_protest_tweets.csv`):**
   - **Coverage:** 417,511 cleaned English tweets spanning February 1, 2021 to March 30, 2021 (58 continuous calendar days).
   - **Role:** Empirical testbed for informal text event extraction, argument parsing, escalation detection, and the India case study.
   - **Scientific Boundary:** Indian agricultural protest tweets are **geographically distinct** from US domestic unrest events in CCC. They are not merged into a fabricated single model; rather, they serve as a rigorous empirical case study evaluating multimodal feature lift on a real protest timeline.

---

## 4. Protected Assets & Integrity Guarantees

The following assets are verified and strictly preserved:
- Existing XGBoost model files: `models/xgboost_1day.json`, `models/xgboost_3day.json`, `models/xgboost_7day.json`.
- Historical baseline benchmark: `results/final_results.csv`.
- Comprehensive test suite: 42 existing tests verified passing.
- Chronological split boundaries: Train (≤ 2022), Validation (2023), Test (2024) with zero target lookahead.

---

## 5. Summary of Completed Improvements

1. **Rich NLP Event Extraction:** Enhanced `backend/social_media_extraction.py` to extract triggers, actors, targets, crowd scale, escalation indicators, and mobilization calls.
2. **Annotation Benchmark:** Created `data/annotation_benchmark.csv` (200 posts stratified across 9 weeks) and `docs/ANNOTATION_GUIDELINES.md` with explicit guidelines separating human labels from model predictions.
3. **NLP-to-Forecasting Integration:** Implemented `backend/social_media_features.py`, `backend/integration_pipeline.py`, and `backend/case_study.py` to aggregate social media signals into daily feature vectors and run comparative forecasting experiments.
4. **Faculty Demonstration Dashboard:** Upgraded `app.py` into a comprehensive 5-tab Streamlit dashboard allowing interactive single-text extraction, temporal browsing, CCC baseline forecasting, case study inspection, and metric evaluation.
5. **Comprehensive Test Suite:** Added `tests/test_integration.py` and expanded `tests/test_nlp.py` covering all new components.
