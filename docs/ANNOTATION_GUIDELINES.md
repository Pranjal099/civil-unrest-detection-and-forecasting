# Annotation Guidelines for Civil Unrest Social Media Benchmark

**Project:** Civil Unrest Event Extraction and Prediction from Social Media Data  
**Repository:** `civil-unrest-detection-and-forecasting`  
**Dataset Scope:** India Farmers' Protest Twitter Corpus (Feb 1 – Mar 30, 2021)  
**Target File:** `data/annotation_benchmark.csv`  

---

## 1. Overview and Purpose

Social media text is informal, noisy, and heavily dominated by political rhetoric, opinions, and hashtag campaigns. A critical scientific challenge in civil unrest event extraction is distinguishing between **actual on-the-ground protest actions / planned mobilizations** versus **passive commentary, solidarity hashtags, or ideological debate**.

This benchmark establishes a standardized, reproducible protocol for evaluating NLP extraction models against human ground truth across seven distinct linguistic and information extraction dimensions.

---

## 2. Benchmark Design and Stratification

The benchmark comprises 200 posts sampled from the 417,511 cleaned English tweets using a **temporal stratified sampling strategy** across all 9 weeks of the protest timeline (Weeks 5 through 13 of 2021).

### Schema of `data/annotation_benchmark.csv`

| Column | Type | Description |
|---|---|---|
| `tweet_id` | Int64 | Unique tweet identifier |
| `text` | String | Original raw social media text |
| `date` | Timestamp | Tweet publication timestamp (UTC) |
| `username` | String | Author handle |
| `event_detected_model` | Int (0/1) | Model-predicted binary event detection |
| `confidence_model` | Float | Model confidence score [0.0 – 1.0] |
| `event_type_model` | String | Model-predicted event typology |
| `trigger_phrase_model` | String | Model-extracted action trigger |
| `actors_model` | JSON List | Model-extracted participant entities |
| `locations_model` | JSON List | Model-extracted geographic entities |
| `target_model` | String | Model-extracted protest target |
| `scale_model` | String | Model-extracted crowd scale |
| `escalation_indicators_model`| JSON List | Model-detected escalation tokens |
| `mobilization_call_model` | Int (0/1) | Model-detected call to action |
| `event_detected_human` | Int (0/1/?) | **Human ground-truth binary label** |
| `event_type_human` | String | **Human ground-truth event category** |
| `annotation_status` | String | `gold_verified`, `reviewed`, or `pending_review` |

---

## 3. Annotation Tasks and Criteria

### Task 1: Binary Event Detection (`event_detected_human`)

Annotators must classify whether the post reports a genuine civil unrest event.

- **Label `1` (Event Detected):**
  - The post reports an ongoing, planned, or recently occurred civil unrest action (protest, demonstration, march, strike, sit-in, blockade, or violent clash).
  - *Example:* `"Farmers blocked highway at Ghazipur border today demanding repeal of farm laws."`
  - *Example:* `"BKU calls for nationwide chakka jam on February 6th from 12pm to 3pm."`

- **Label `0` (No Event):**
  - The post expresses personal opinions, solidarity, prayers, quotes, or general political criticism without reporting an actionable event or specific mobilization.
  - *Example:* `"Supporting farmers means supporting our country #StopHateAgainstFarmers"`
  - *Example:* `"Neutrality helps the oppressor, never the victim. Speak up for farmers."`
  - *Example:* `"Great read to gain a better understanding of the religious angle."`

- **Label `?` (Uncertain / Ambiguous):**
  - Use when the text cannot be determined without external context (e.g. metaphorical calls like *"we will fight in elections"*).

---

### Task 2: Event Typology Classification (`event_type_human`)

If `event_detected_human == 1`, assign one primary category:

1. **`protest`**: General gathering or demonstration voicing political/economic dissent.
2. **`demonstration`**: Structured march, procession, or public rally.
3. **`strike`**: Labor work stoppage, complete shutdown (*bandh*), or commercial boycott.
4. **`riot`**: Violent confrontation involving stone-pelting, arson, physical fighting, or heavy police clash.
5. **`labor dispute`**: Specific employer-employee grievances or wage negotiations.
6. **`political unrest`**: Defiance of state directives, section 144 defiance, or constitutional agitation.
7. **`other`**: Unrest forms not fitting above (e.g. hunger strikes, tractor parades).

---

### Task 3: Event Argument Extraction

When reviewing model extractions:

- **Trigger Phrase:** The core verb or verbal phrase denoting the unrest action (e.g. *protested*, *held rally*, *blocked highway*, *chakka jam*).
- **Actor / Organizer:** Group or individual leading or enacting the protest (e.g. *BKU*, *Samyukt Kisan Morcha*, *Transport union*, *farmers*).
- **Location:** Geographic anchors, borders, cities, or landmarks (e.g. *Singhu border*, *Tikri*, *Delhi*, *Punjab*).
- **Target:** Entity, legislation, or institution opposed (e.g. *farm laws*, *Central Government*, *corporate buyers*).
- **Scale:** Numeric or qualitative crowd indicators (e.g. *thousands*, *50,000*, *massive crowd*).
- **Escalation Indicators:** Presence of tactical conflict elements: *violence, clash, teargas, water cannon, lathi charge, arrest, detention, barricade, blockade, arson*.
- **Mobilization Call:** Presence of explicit action-solicitation language: *join us, march to, assemble at, be there, support farmers, chalo*.

---

## 4. Evaluation Metrics Formulation

The evaluation script `backend/nlp_evaluation.py` evaluates model extractions against human annotations using standard micro- and macro-averaged metrics:

$$\text{Precision} = \frac{\text{TP}}{\text{TP} + \text{FP}}, \quad \text{Recall} = \frac{\text{TP}}{\text{TP} + \text{FN}}, \quad \text{F1} = 2 \cdot \frac{\text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}}$$

For argument extraction (Locations, Actors, Triggers), token-level exact match and relaxed span overlaps are evaluated.
