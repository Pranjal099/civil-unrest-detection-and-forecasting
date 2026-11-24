# Dataset Documentation

This repository manages multi-source datasets for civil unrest event extraction and predictive time-series forecasting.

---

## 1. Tracked Sample & Case Study Data

The following curated tabular datasets are tracked in the repository:

| File | Records | Purpose | Key Columns |
| :--- | :--- | :--- | :--- |
| `demo_social_posts.csv` | 10 | Synthetic and sample social posts for pipeline demo and testing | `id`, `text` |
| `protest_articles.csv` | 91 | Raw news article headlines retrieved via Google News RSS feeds | `event`, `text`, `source`, `url`, `published_at`, `collected_at` |
| `cleaned_protest_articles.csv` | 84 | Deduplicated and keyword-filtered protest headlines | `event`, `text`, `source`, `url`, `published_at`, `relevant` |
| `final_protest_articles.csv` | 47 | Verified headlines strictly within historical protest date windows | `event`, `text`, `source`, `url`, `published_at` |

### Case Study Events
- **Nepal Gen Z Protests (2025)**: Window `2025-09-01` to `2025-10-31`
- **Bangladesh Student Uprising (2024)**: Window `2024-07-01` to `2024-09-30`
- **India Farmers' Protest (2020–2021)**: Window `2020-11-01` to `2022-01-31`

---

## 2. Large External Datasets (Git-Ignored)

Due to file size and licensing constraints, the following raw and processed corpora are omitted from version control:

### A. Crowd Counting Consortium (CCC) Compiled Dataset (`ccc_compiled_20212024.csv`)
* **Size**: ~113 MB (CSV, Latin-1 encoded)
* **Coverage**: US nationwide civil unrest, protest, and demonstration records from 2021-01-01 through 2024-12-31.
* **Granularity**: Event-level records aggregated into daily state-level counts across all 50 states + DC (74,511 state-day observations).
* **Source**: Crowd Counting Consortium (Harvard University / University of Connecticut).

### B. Farmers' Protest Tweets Corpus
* **Raw JSON**: `farmers-protest-tweets-2021-03-5.json` (~1.3 GB, line-delimited JSON).
* **Parsed CSV**: `farmers_protest_tweets.csv` (~120 MB).
* **Cleaned CSV**: `cleaned_farmers_protest_tweets.csv` (~120 MB, English-only, deduplicated).
* **Streaming Processor**: Ingested via `backend/process_farmers_tweets.py` in 10,000-record chunks.

---

## 3. Data Governance & Reproduction

To replicate the training and evaluation pipelines:
1. Obtain `ccc_compiled_20212024.csv` and place it in `data/`.
2. Run `make evaluate` to reproduce 2024 test metrics using the trained XGBoost models.
3. Use `backend/train.py --dry-run` to verify dataset loading and feature engineering without mutating saved models.
