# Benchmark & Baseline Analysis

This directory documents baseline benchmarks and comparative references for the Civil Unrest Event Extraction and Prediction project.

---

## 1. Reference Baseline: JHU-CUT (Civil Unrest on Twitter)

We compare our pipeline design against the benchmark established in:
> **Alexandra DeLucia et al.**, *"Civil Unrest on Twitter (CUT): A Dataset of Tweets to Support Research on Civil Unrest"*, EMNLP 2020 Workshop on Noisy User-generated Text (W-NUT).

### JHU-CUT Baseline Architecture
The CUT benchmark evaluated tweet-level civil unrest classification using:
- **Keyword Count Model**: F1 = 0.782
- **Unigram Count Model**: F1 = 0.775
- **BERTweet Transformer Model**: F1 = 0.814

---

## 2. Architectural Comparison

| Dimension | JHU-CUT Baseline | Our Pipeline Architecture |
| :--- | :--- | :--- |
| **Input Modality** | Tweet text with ID lookups | Multi-source (Social posts, streaming tweets, Google News RSS) |
| **NLP Methodology** | Supervised BERTweet / N-gram | Two-stage Zero-Shot classification (BART Large MNLI) + SpaCy NER |
| **Event Typology** | Binary (unrest vs non-unrest) | 7-category taxonomy (protest, demonstration, strike, riot, labor, political, other) |
| **Entity Extraction** | None | Automatic GPE/LOC, ORG, DATE extraction |
| **Predictive Modeling** | None (tweet-level classification only) | Multi-horizon state-level time-series forecasting (XGBoost) |
| **Forecast Horizons** | N/A | **1-day** (PR-AUC 0.919), **3-day** (PR-AUC 0.976), **7-day** (PR-AUC 0.994) |

---

## 3. Directory Structure

```
benchmark/
├── README.md               # This comparative overview
└── CUT/                    # (Git-ignored) JHU-CUT reference scripts and data schema
    ├── BERTweet_utils.py
    ├── bertweet_model.py
    ├── ngram_model.py
    └── ...
```
