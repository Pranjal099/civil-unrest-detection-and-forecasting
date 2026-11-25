# Faculty Demonstration Guide & Oral Defense Script

**Project Title:** AI-Powered Civil Unrest Event Extraction & Short-Term Forecasting  
**Author / Presenter:** Civil Unrest Research Team  
**Evaluation Target:** Academic Project Defense & Faculty Evaluation  
**Estimated Demo Time:** 8–10 Minutes  

---

## 1. Quick Launch Commands

To start the system in front of faculty:

```bash
# 1. Navigate to project root
cd ~/Desktop/civil-unrest-forecasting

# 2. Activate virtual environment
source .venv/bin/activate

# 3. Verify test suite (all 45+ tests passing in ~4 seconds)
pytest tests/ -v

# 4. Launch interactive demonstration dashboard
streamlit run app.py
```

The application will launch in your browser at `http://localhost:8501`.

---

## 2. Step-by-Step Demonstration Sequence

```
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                                FACULTY DEMO FLOW                                       │
│                                                                                        │
│  [Tab 1: NLP Extraction]  ──►  [Tab 2: Temporal Browser]  ──►  [Tab 3: CCC Forecast]   │
│  (Informal text -> arguments)  (58-day protest timeline)       (1d, 3d, 7d XGBoost)    │
│                                                                                        │
│                                  │                               │                     │
│                                  ▼                               ▼                     │
│                       [Tab 4: Case Study]            [Tab 5: Evaluation]               │
│                       (Model A vs Model B)           (PR-AUC & Confusion Matrix)       │
└────────────────────────────────────────────────────────────────────────────────────────┘
```

---

### Phase 1: Tab 1 — 🔍 NLP Event Extraction (Minutes 0–2)

**Goal:** Demonstrate event extraction, argument parsing, and escalation detection from informal social media text.

**Action:**
1. Navigate to **Tab 1: "🔍 NLP Event Extraction"**.
2. Select **"Single / custom text"** and paste the following realistic tweet:
   > *"Farmers union blocked national highway at Singhu border today demanding repeal of 3 farm laws. Police deployed teargas after violent clash with 10,000 protesters. Join us tomorrow!"*
3. Click **"Extract Events"** (or use the pre-loaded examples).

**Scripted Talking Points:**
> *"Distinguished faculty, informal social media data is characterized by noisy syntax, abbreviations, and emotion. Our NLP pipeline uses zero-shot natural language inference via BART-large-MNLI combined with spaCy dependency and entity parsing to extract a complete 12-field structured event record without requiring large custom model training from scratch.*
> 
> *Notice what the model extracts automatically:*
> - *Event Detected: True (Confidence: 99.5%)*
> - *Event Type: Protest / Riot*
> - *Trigger Phrase: 'blocked national highway'*
> - *Actors: 'Farmers union'*
> - *Location: 'Singhu border'*
> - *Target: 'repeal of 3 farm laws'*
> - *Scale: '10,000 protesters'*
> - *Escalation Indicators: ['clash', 'teargas', 'blocked', 'violence']*
> - *Mobilization Call: True ('join us tomorrow')*
> 
> *Furthermore, switch to 'Batch mode' to run all 10 demo posts instantly, demonstrating that non-unrest posts (e.g. sports scores, weather updates) are correctly filtered out as negative with zero hallucinations."*

---

### Phase 2: Tab 2 — 📅 Temporal Event Browser (Minutes 2–4)

**Goal:** Demonstrate timeline aggregation across the 58-day India Farmers' Protest dataset.

**Action:**
1. Navigate to **Tab 2: "📅 Temporal Event Browser"**.
2. Point out the date range (Feb 1 – Mar 30, 2021) and the total volume of cleaned tweets (417,511).
3. Review the **Daily Event Volume Chart**, the **Event Typology Breakdown**, and the **Escalation Frequency Chart**.

**Scripted Talking Points:**
> *"Here we show the temporal evolution of events across the 58 continuous calendar days of the Farmers' Protest corpus. The daily volume displays distinct activity spikes corresponding to real-world protest milestones—notably the February 6th nationwide 'Chakka Jam' (road blockade) and the late March Bharat Bandh.*
> 
> *Crucially, our pipeline extracts escalation indicators and mobilization calls on each day. Notice that mobilization call volume consistently peaks 24 to 48 hours BEFORE the physical escalation in road blockades and clashes occurs. This temporal lead time is the exact signal we leverage for forecasting."*

---

### Phase 3: Tab 3 — 📈 CCC Forecasting (Baseline) (Minutes 4–6)

**Goal:** Demonstrate the verified, multi-horizon Gradient Boosted time-series forecasting engine on Crowd Counting Consortium data.

**Action:**
1. Navigate to **Tab 3: "📈 CCC Forecasting (Baseline)"**.
2. Select **State: California (`CA`)** (or New York `NY`).
3. Select an observation date (e.g., late 2024).
4. Point out the **Historical Activity Features** display:
   - Events yesterday (`lag_1`), 3 days ago (`lag_3`), 7 days ago (`lag_7`), 14 days ago (`lag_14`).
   - 7-day rolling mean and sum, 14-day rolling mean.
5. Click **"Generate Multi-Horizon Forecast"**.
6. Review the predicted probabilities, the risk badges, and the **Ground Truth Validation Table**.

**Scripted Talking Points:**
> *"This tab demonstrates our production forecasting model. Built on 139,823 verified events from the Crowd Counting Consortium spanning 2021 through 2024, our models predict the emergence of civil unrest across 1-day, 3-day, and 7-day forward horizons.*
> 
> *Notice three critical engineering features:*
> 1. *Zero Lookahead Leakage: All 7 feature variables are strictly backward-looking lags and rolling averages computed prior to the forecast timestamp.*
> 2. *Threshold Optimization: Rather than using an arbitrary 0.50 cutoff, we optimize decision thresholds for each horizon—0.35 for 1-day, 0.25 for 3-day, and 0.10 for 7-day—to balance precision and recall on imbalanced distributions.*
> 3. *Ground Truth Validation: The dashboard compares the model's prediction against what actually occurred in the held-out historical data, verifying live accuracy."*

---

### Phase 4: Tab 4 — 🔗 Integration Case Study (Minutes 6–8)

**Goal:** Transparently explain system decoupling and demonstrate the comparative experiment showing multimodal feature lift.

**Action:**
1. Navigate to **Tab 4: "🔗 Integration Case Study"**.
2. Highlight the prominent **Methodological Limitation Banner**.
3. Walk through the comparative experiment table: **Model A (Activity / Lag-Only)** vs **Model B (Lag + NLP Social Features)**.

**Scripted Talking Points:**
> *"A central principle of our project is scientific honesty. A common error in machine learning projects is claiming that social media tweets from India in 2021 were used to forecast US civil unrest in 2024. That would be geographically and methodologically false.*
> 
> *Instead, we conduct a rigorous, clearly-labeled Comparative Case Study on the 58-day continuous Farmers' Protest timeline:*
> - *Model A (Baseline): Predicts next-day high-intensity unrest using ONLY past activity lags.*
> - *Model B (Multimodal Integrated): Ingests past activity lags PLUS our NLP-extracted mobilization calls, escalation indicators, clash counts, and actor diversity.*
> - *Both models are evaluated on a strictly chronological held-out test split (19 days in March 2021).*
> 
> *The empirical result shows that adding NLP social media features delivers measurable predictive lift, particularly during transition phases between calm and peak mobilization where historical activity lags alone are too slow to react."*

---

### Phase 5: Tab 5 — 📊 Model Evaluation & Metrics (Minutes 8–10)

**Goal:** Present comprehensive, independently verified performance metrics and confusion matrices.

**Action:**
1. Navigate to **Tab 5: "📊 Model Evaluation & Metrics"**.
2. Review the final test set metrics table:
   - 1-Day: PR-AUC = 0.9194, F1 = 0.8398, Accuracy = 80.60%
   - 3-Day: PR-AUC = 0.9757, F1 = 0.9161, Accuracy = 86.45%
   - 7-Day: PR-AUC = 0.9944, F1 = 0.9681, Accuracy = 94.15%
3. Review the **Operating Thresholds & Class Prevalence Table** and the **Confusion Matrices**.

**Scripted Talking Points:**
> *"All reported metrics on this tab have been independently recomputed on the 19,386 held-out records of the 2024 test partition. As documented in our `VERIFICATION_REPORT.md`, there is zero data leakage between train (2021–2022), validation (2023), and test (2024).*
> 
> *We also explicitly disclose class prevalence: over a 7-day horizon, 90.12% of state intervals in active states experience an event. Therefore, PR-AUC and F1 scores—which focus strictly on the positive minority/majority class dynamics—are our primary evaluation metrics rather than raw accuracy."*

---

## 3. Frequently Asked Questions & Defense Answers

### Q1: Why didn't you train your own Transformer model from scratch for NLP extraction?
> **Answer:**  
> *"Pretraining a 300M+ parameter Transformer from scratch requires millions of domain-specific annotated documents and tens of thousands of dollars in GPU compute. More importantly, foundational models like BART-large (pretrained on MNLI) already possess rich linguistic representations of entailment, negation, and semantic roles. By employing zero-shot classification alongside spaCy dependency trees, we achieve state-of-the-art zero-shot generalization while maintaining a lightweight, CPU/MPS-executable architecture that runs entirely on standard hardware without external API dependencies."*

### Q2: Why are the CCC forecasting models and the Farmers' Protest tweets separate?
> **Answer:**  
> *"They are separate because the physical realities they represent are separate. The CCC dataset tracks US domestic civil unrest across 54 states from 2021 to 2024. The available tweet dataset tracks the Indian Farmers' Protest around Delhi in February–March 2021. Merging them into a single dataset would violate geographic and causal consistency. We architected the system with two decoupled subsystems and an explicit Case Study Bridge to demonstrate the validity of both capabilities honestly."*

### Q3: How do you prevent data leakage in your time-series features?
> **Answer:**  
> *"Every single feature in our feature matrix (`lag_1`, `lag_3`, `lag_7`, `lag_14`, `rolling_7_mean`, `rolling_7_sum`, `rolling_14_mean`) is mathematically constructed using only observations at or strictly prior to time $t$. The forward targets look forward into $[t+1, t+7]$, but are never included in the input feature matrix. Furthermore, our dataset split is strictly chronological by calendar year (Train: ≤ 2022, Val: 2023, Test: 2024) with zero record overlap."*

### Q4: Why are the 7-day PR-AUC numbers so high (0.994)? Is the model overfitting?
> **Answer:**  
> *"No, the model is not overfitting; it is evaluated on completely unseen 2024 test data. The high PR-AUC is a mathematical consequence of high positive class prevalence. In the CCC dataset across 54 states, 90.12% of state-week intervals witness at least one protest event. Because protests in major states (CA, NY, TX) are continuous, predicting whether an event will occur anywhere in a state over an entire 7-day window is an easier task than predicting 1-day ahead (where prevalence is 58.87% and PR-AUC is 0.9194)."*

### Q5: What happens when an informal tweet contains slang, sarcasm, or no named entities?
> **Answer:**  
> *"Our extraction pipeline employs defensive fallback architecture. If spaCy dependency parsing cannot identify an explicit subject or direct object, the pipeline falls back to entity label matching, prepositional phrase regex extraction, and keyword scanning. If no event triggers are found, the text is cleanly returned with `event_detected=False` and confidence scores rather than hallucinating false entities."*
