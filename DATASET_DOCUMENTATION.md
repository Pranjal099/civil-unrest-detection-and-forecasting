# Dataset Documentation & Provenance

**Project Title:** AI-Powered Civil Unrest Event Extraction & Forecasting  
**Repository:** `civil-unrest-detection-and-forecasting`  
**Last Updated:** 2026-11-25  

---

## 1. Corpus Summary Table

| Dataset Identifier | Primary Role | Geographic Scope | Temporal Range | Record Count | Disk Size | Version Control Status |
|---|---|---|---|---|---|---|
| **Crowd Counting Consortium (CCC)** | Time-Series Forecasting | United States (54 states/territories) | 2021-01-01 to 2024-12-24 | 139,823 raw events &rarr; 77,760 state-days | 113 MB | **Ignored** (`.gitignore`) |
| **India Farmers' Protest Tweets** | NLP Extraction & Case Study | India (Delhi NCR, Punjab, Haryana) | 2021-02-01 to 2021-03-30 | 417,511 cleaned tweets (1.2M raw lines) | 120 MB (CSV) / 1.3 GB (JSON) | **Ignored** (`.gitignore`) |
| **Annotation Benchmark** | NLP Extraction Evaluation | India Farmers' Protest sample | 2021-02-01 to 2021-03-30 | 200 stratified posts | ~65 KB | **Tracked** in Git |
| **Demo Social Posts** | Quick Interactive Smoke Testing | Synthetic multi-domain unrest | N/A (illustrative) | 10 curated posts | < 1 KB | **Tracked** in Git |
| **Temporal Extraction Sample** | Case Study Feature Construction | India Farmers' Protest sample | 2021-02-01 to 2021-03-30 | 2,900 posts (50 posts/day) | ~1.1 MB | **Tracked** in Git |

---

## 2. Crowd Counting Consortium (CCC) Dataset

### 2.1 Provenance and Acquisition
The primary time-series forecasting corpus is derived from the **Crowd Counting Consortium (CCC)**, an academic research initiative jointly directed by scholars at Harvard Kennedy School and the University of Connecticut. CCC systematically aggregates and cross-verifies public reports of political crowds, demonstrations, strikes, marches, and civil resistance actions across the United States.

- **Local Path:** `data/ccc_compiled_20212024.csv`
- **Encoding:** Latin-1 (`ISO-8859-1`)
- **Raw Event Count:** 139,823 recorded events.

### 2.2 Processing and State-Day Matrix Construction
In `backend/data_processing.py` (`load_and_process_data()`), raw event entries are aggregated into a daily panel matrix:
1. **Full Cartesian Grid:** A complete date grid is generated for each of the 54 US states and territories across the 4-year span (January 1, 2021 through December 24, 2024), yielding exactly **77,760 state-day records**.
2. **Event Count Aggregation:** Daily unrest occurrence is summed into `event_count`. Days with zero recorded unrest are explicitly preserved as zero-activity observations to prevent selection bias.

### 2.3 Feature Matrix & Target Definitions
To guarantee zero lookahead leakage, all engineered features are strictly backward-looking:
- **Lag Features:** `lag_1`, `lag_3`, `lag_7`, `lag_14` (event counts observed $t-1, t-3, t-7, t-14$ days prior).
- **Rolling Aggregations:**
  - `rolling_7_mean`: 7-day backward moving average.
  - `rolling_7_sum`: 7-day backward total events.
  - `rolling_14_mean`: 14-day backward moving average.
- **Forward Forecasting Targets:**
  - `target_1day`: Binary flag indicating whether $\ge 1$ unrest event occurs at $t+1$.
  - `target_3day`: Binary flag indicating whether $\ge 1$ unrest event occurs in the window $[t+1, t+3]$.
  - `target_7day`: Binary flag indicating whether $\ge 1$ unrest event occurs in the window $[t+1, t+7]$.

### 2.4 Class Distribution & Operating Thresholds

| Horizon | Positive Event Prevalence | Optimized Classification Threshold | Test Set Positive Count (2024) | Test Set Negative Count (2024) |
|---|---|---|---|---|
| **1-day** | **58.87%** | **0.35** | 11,413 | 7,973 |
| **3-day** | **77.94%** | **0.25** | 15,109 | 4,277 |
| **7-day** | **90.12%** | **0.10** | 17,471 | 1,915 |

> **Methodological Note on Class Imbalance:**  
> In populous US states (e.g. California, New York, Texas), civil unrest events occur very frequently. Over a 7-day forward window, 90.12% of state-week intervals witness at least one protest event. Consequently, high PR-AUC values (0.919 to 0.994) reflect both strong model discrimination and high baseline positive class prevalence.

### 2.5 Chronological Partitioning

| Partition | Date Range | Calendar Days | Record Count | Overlap with Other Splits |
|---|---|---|---|---|
| **Train** | 2021-01-15 &rarr; 2022-12-31 | 716 days | 38,664 state-days | **0 rows (0.0%)** |
| **Validation** | 2023-01-01 &rarr; 2023-12-31 | 365 days | 19,710 state-days | **0 rows (0.0%)** |
| **Test** | 2024-01-01 &rarr; 2024-12-24 | 359 days | 19,386 state-days | **0 rows (0.0%)** |

---

## 3. India Farmers' Protest Twitter Corpus

### 3.1 Provenance and Scope
The social media extraction corpus originates from the widely studied **2020–2021 Indian Farmers' Protest** Twitter archive. During early 2021, hundreds of thousands of agricultural workers established encampments along the borders of Delhi (Singhu, Tikri, Ghazipur) protesting three deregulation farm bills passed by the Parliament of India.

- **Raw Archive:** `data/farmers-protest-tweets-2021-03-5.json` (1.3 GB JSON).
- **Cleaned Dataset:** `data/cleaned_farmers_protest_tweets.csv` (120 MB CSV).
- **Row Count:** 417,511 cleaned English tweets (1,232,827 rows before deduplication).
- **Temporal Span:** February 1, 2021 (00:00:07 UTC) through March 30, 2021 (03:33:46 UTC) — exactly **58 continuous calendar days**.
- **Daily Volume:** Average of 7,198 tweets/day (ranging from 576 to 83,403 tweets/day).

### 3.2 Geographic & Subsystem Decoupling
Because the Farmers' Protest occurred in northern India during February–March 2021, and the CCC dataset tracks domestic events in the United States from 2021–2024:
1. **The social media dataset was NOT used to train the production US CCC XGBoost models.**
2. **Indian tweets are NOT merged into US state observations.**
3. The dataset serves two specific, scientifically legitimate purposes:
   - As an informal-text NLP benchmark for evaluating zero-shot event extraction, argument parsing, escalation detection, and mobilization classification.
   - As an empirical single-event case study testing whether adding NLP social-media features provides predictive lift over historical activity lags alone.

---

## 4. Evaluation and Benchmark Subsets

### 4.1 Annotation Benchmark (`data/annotation_benchmark.csv`)
- **Size:** 200 tweets sampled across all 9 weeks of the protest timeline.
- **Sampling:** Stratified by calendar week (Weeks 5 to 13 of 2021).
- **Schema:** Contains both model-generated extraction fields and empty human ground-truth fields (`event_detected_human`, `event_type_human`, `annotation_status`).
- **Gold-Standard Subset:** 50 rows labeled with verified ground truth following `docs/ANNOTATION_GUIDELINES.md` for immediate reproducible evaluation.

### 4.2 Demo Social Posts (`data/demo_social_posts.csv`)
- **Size:** 10 curated posts.
- **Purpose:** Used for dashboard demonstration and rapid regression smoke testing.
