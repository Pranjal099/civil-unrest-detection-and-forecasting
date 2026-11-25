import streamlit as st
import pandas as pd
import json
from pathlib import Path

from backend.predict import predict_unrest, FEATURES
from backend.data_processing import load_and_process_data

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Civil Unrest Forecasting",
    page_icon="📊",
    layout="wide"
)

# ---------------- CUSTOM CSS ----------------
st.markdown("""
<style>
.stApp {
    background-color: #F8FAFC;
    color: #0F172A;
}

.block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
}

[data-testid="stSidebar"] {
    background-color: #0F172A;
}

[data-testid="stSidebar"] * {
    color: #F8FAFC !important;
}

.subtitle {
    color: #475569;
    font-size: 1.05rem;
    margin-bottom: 1.5rem;
}
</style>
""", unsafe_allow_html=True)

ROOT = Path(__file__).parent


# ---------------- CACHED DATA LOADERS ----------------
@st.cache_data(show_spinner="Loading historical event database...")
def get_historical_data():
    try:
        return load_and_process_data()
    except FileNotFoundError:
        return None
    except Exception:
        return None


@st.cache_data(show_spinner="Loading sample extractions...")
def load_farmers_extractions():
    p = ROOT / "results" / "farmers_protest_sample_extractions.csv"
    if p.exists():
        return pd.read_csv(p)
    return None


@st.cache_data(show_spinner="Loading final results...")
def load_final_results():
    p = ROOT / "results" / "final_results.csv"
    if p.exists():
        return pd.read_csv(p)
    return None


@st.cache_data(show_spinner="Loading demo posts...")
def load_demo_posts():
    p = ROOT / "data" / "demo_social_posts.csv"
    if p.exists():
        return pd.read_csv(p)
    return None


def load_case_study():
    p = ROOT / "results" / "case_study_results.json"
    if p.exists():
        try:
            with open(p) as f:
                return json.load(f)
        except Exception:
            return None
    return None


# ---------------- SIDEBAR ----------------
with st.sidebar:
    st.markdown("## Civil Unrest")
    st.markdown("### Forecasting System")
    st.divider()

    st.markdown("**Model Information**")
    st.markdown("- **Algorithm**: Gradient Boosted Trees (XGBoost)")
    st.markdown("- **Horizons**: 1, 3, and 7 Days Ahead")
    st.markdown("- **Reference Corpus**: CCC Compiled (2021–2024)")
    st.markdown("- **Decision Thresholds**: 0.35 (1-d), 0.25 (3-d), 0.10 (7-d)")
    st.divider()

    st.caption("Academic ML & Information Extraction Project")
    st.caption("State-level historical activity forecasting")


# ---------------- HEADER ----------------
st.title("Civil Unrest Event Forecasting Dashboard")
st.markdown(
    '<p class="subtitle">'
    'Predictive early-warning indicators of civil unrest based on historical temporal event patterns.'
    '</p>',
    unsafe_allow_html=True
)

# ---------------- TABS ----------------
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🔍 NLP Event Extraction",
    "📅 Temporal Event Browser",
    "📈 CCC Forecasting (Baseline)",
    "🔗 Integration Case Study",
    "📊 Model Evaluation & Metrics",
])


# ═══════════════════════════════════════════════════════════
# TAB 1 — NLP Event Extraction
# ═══════════════════════════════════════════════════════════
with tab1:
    st.header("NLP Event Extraction")
    st.caption(
        "Paste social media text below to extract structured civil unrest event features "
        "using the zero-shot NLP pipeline."
    )

    DEFAULT_TEXTS = (
        "Farmers blocked highways across Punjab today demanding MSP guarantees.\n"
        "Students clashed with police outside the education ministry. Teargas deployed.\n"
        "Workers announce indefinite strike at steel plant; thousands expected to join."
    )

    mode = st.segmented_control(
        "Input mode",
        options=["Single / custom text", "Batch (demo_social_posts.csv)"],
        default="Single / custom text",
    )

    if mode == "Single / custom text":
        user_text = st.text_area(
            "Social media text",
            value=DEFAULT_TEXTS,
            height=140,
            label_visibility="collapsed",
        )
        texts_to_run = [t.strip() for t in user_text.strip().splitlines() if t.strip()]
    else:
        demo_df = load_demo_posts()
        if demo_df is not None and "text" in demo_df.columns:
            texts_to_run = demo_df["text"].dropna().tolist()
            st.info(f"Loaded {len(texts_to_run)} posts from `data/demo_social_posts.csv`.")
        else:
            st.warning("`data/demo_social_posts.csv` not found or missing `text` column.")
            texts_to_run = []

    if st.button(":material/search: Extract Events", type="primary"):
        if not texts_to_run:
            st.warning("No text to process.")
        else:
            # Lazy-load NLP module only on click
            try:
                from backend.social_media_extraction import extract_event  # noqa: PLC0415
                extract_ok = True
            except Exception as import_err:
                st.warning(
                    f"NLP models are not available: `{import_err}`. "
                    "Ensure spaCy and transformers are installed."
                )
                extract_ok = False

            if extract_ok:
                rows = []
                with st.spinner(f"Extracting events from {len(texts_to_run)} text(s)…"):
                    for txt in texts_to_run:
                        try:
                            result = extract_event(txt)
                        except Exception as exc:
                            result = {
                                "event_detected": False,
                                "confidence": 0.0,
                                "event_type": "error",
                                "trigger_phrase": None,
                                "actors": [],
                                "locations": [],
                                "escalation_indicators": [],
                                "mobilization_call": False,
                                "scale": None,
                            }
                            st.error(f"Error on: `{txt[:60]}…` → {exc}")
                        rows.append({"text": txt[:80], **result})

                st.success(f"Extracted {len(rows)} result(s).")

                for i, row in enumerate(rows):
                    with st.container(border=True):
                        detected = row.get("event_detected", False)
                        badge_color = "green" if detected else "red"
                        badge_label = "Event detected" if detected else "No event"

                        c1, c2 = st.columns([3, 1])
                        with c1:
                            st.markdown(f"**Text {i+1}:** {row['text']}")
                        with c2:
                            st.badge(badge_label, color=badge_color)

                        cols = st.columns(4)
                        cols[0].metric("Confidence", f"{row.get('confidence', 0):.2f}")
                        cols[1].metric("Event type", row.get("event_type") or "—")
                        cols[2].metric("Trigger phrase", row.get("trigger_phrase") or "—")
                        cols[3].metric("Scale", row.get("scale") or "—")

                        actors = row.get("actors") or []
                        locs = row.get("locations") or []
                        esc = row.get("escalation_indicators") or []
                        mob = row.get("mobilization_call", False)

                        st.markdown(
                            f"**Actors:** {', '.join(actors) if actors else '—'}  \n"
                            f"**Locations:** {', '.join(locs) if locs else '—'}  \n"
                            f"**Mobilization call:** {'✅ Yes' if mob else '❌ No'}"
                        )

                        if esc:
                            for indicator in esc:
                                st.badge(f"⚠️ {indicator}", color="orange")


# ═══════════════════════════════════════════════════════════
# TAB 2 — Temporal Event Browser
# ═══════════════════════════════════════════════════════════
with tab2:
    st.header("Temporal Event Browser — Farmers' Protest Case Study")

    st.warning(
        "This dataset covers **ONLY** the India Farmers' Protest (Feb–Mar 2021). "
        "It does **not** represent global civil unrest.",
        icon="⚠️",
    )

    df_ext = load_farmers_extractions()

    if df_ext is None:
        st.info(
            "`results/farmers_protest_sample_extractions.csv` has not been generated yet. "
            "Run the NLP worker to produce this file.",
            icon=":material/schedule:",
        )
    else:
        # Summary metrics
        total = len(df_ext)
        date_col = next(
            (c for c in df_ext.columns if "date" in c.lower()), None
        )
        event_col = next(
            (c for c in df_ext.columns if "event_detected" in c.lower()), None
        )

        detected_count = int(df_ext[event_col].sum()) if event_col else 0
        pct_detected = (detected_count / total * 100) if total else 0

        m1, m2, m3 = st.columns(3)
        m1.metric("Total tweets analyzed", f"{total:,}")
        m2.metric("Detected as events", f"{detected_count:,}")
        m3.metric("Detection rate", f"{pct_detected:.1f}%")

        if date_col:
            df_ext[date_col] = pd.to_datetime(df_ext[date_col], errors="coerce")
            min_d = df_ext[date_col].min()
            max_d = df_ext[date_col].max()
            st.caption(f"**Date range covered:** {min_d.date()} → {max_d.date()}")

            # Event counts by date bar chart
            st.subheader("Event counts by date")
            if event_col:
                date_grp = (
                    df_ext[df_ext[event_col].astype(bool)]
                    .groupby(df_ext[date_col].dt.date)
                    .size()
                    .rename("Event count")
                )
                st.bar_chart(date_grp, alt="Bar chart of event counts by date")
            else:
                daily = df_ext.groupby(df_ext[date_col].dt.date).size().rename("Tweet count")
                st.bar_chart(daily, alt="Bar chart of tweet volume by date")

        # Event type distribution
        type_col = next(
            (c for c in df_ext.columns if "event_type" in c.lower()), None
        )
        if type_col:
            st.subheader("Event type distribution")
            type_counts = df_ext[type_col].dropna().value_counts().rename("Count")
            st.bar_chart(type_counts, alt="Bar chart of event type distribution")

        # Escalation indicator frequency
        esc_col = next(
            (c for c in df_ext.columns if "escalation" in c.lower()), None
        )
        if esc_col:
            st.subheader("Escalation indicator frequency")
            all_indicators = []
            for val in df_ext[esc_col].dropna():
                if isinstance(val, str) and val.strip():
                    try:
                        items = json.loads(val.replace("'", '"'))
                        if isinstance(items, list):
                            all_indicators.extend(items)
                    except Exception:
                        all_indicators.extend(
                            [x.strip() for x in val.strip("[]").split(",") if x.strip()]
                        )
            if all_indicators:
                esc_series = pd.Series(all_indicators).value_counts().rename("Count")
                st.bar_chart(esc_series, alt="Bar chart of escalation indicator frequency")
            else:
                st.info("No escalation indicators found in the data.")


# ═══════════════════════════════════════════════════════════
# TAB 3 — CCC Forecasting (Baseline)  [original tab verbatim]
# ═══════════════════════════════════════════════════════════
with tab3:
    st.header("CCC Forecasting (Baseline)")

    data = get_historical_data()

    if data is None:
        st.warning(
            "Historical CCC dataset (`data/ccc_compiled_20212024.csv`) was not found. "
            "Place the compiled dataset in `data/` to enable interactive historical lookups. "
            "The model inference service remains operational via `backend.predict`."
        )
    else:
        # State and date selection
        with st.container(border=True):
            st.subheader("Select historical region and observation date")
            col1, col2 = st.columns(2)

            available_states = sorted(data["state"].unique())

            with col1:
                selected_state = st.selectbox(
                    "Select State / Region",
                    available_states,
                    index=available_states.index("CA") if "CA" in available_states else 0,
                    key="tab3_state",
                )

            state_data = data[data["state"] == selected_state].copy()
            min_date = state_data["date"].min().date()
            max_date = state_data["date"].max().date()

            with col2:
                selected_date = st.date_input(
                    "Select Observation Date",
                    value=max_date,
                    min_value=min_date,
                    max_value=max_date,
                    key="tab3_date",
                )

        selected_row = state_data[state_data["date"].dt.date == selected_date]

        if not selected_row.empty:
            row = selected_row.iloc[0]
            features = {feat: float(row[feat]) for feat in FEATURES}

            st.subheader("Historical activity features")
            st.caption("Features extracted from historical event time-series up to observation date.")

            feature_cols = st.columns(4)
            feature_display = [
                ("Events yesterday", "lag_1"),
                ("Events 3 days ago", "lag_3"),
                ("Events 7 days ago", "lag_7"),
                ("Events 14 days ago", "lag_14"),
                ("7-day rolling average", "rolling_7_mean"),
                ("7-day total events", "rolling_7_sum"),
                ("14-day rolling average", "rolling_14_mean"),
            ]

            for i, (label, key) in enumerate(feature_display):
                with feature_cols[i % 4]:
                    with st.container(border=True):
                        st.metric(label, f"{features[key]:.2f}")

            st.write("")
            if st.button("Generate Multi-Horizon Forecast", type="primary", width="stretch", key="tab3_forecast"):
                try:
                    results = {
                        1: predict_unrest(features, 1),
                        3: predict_unrest(features, 3),
                        7: predict_unrest(features, 7),
                    }

                    st.divider()
                    st.subheader("Forecast results")
                    result_cols = st.columns(3)

                    for col, horizon in zip(result_cols, [1, 3, 7]):
                        res = results[horizon]
                        with col:
                            with st.container(border=True):
                                st.metric(
                                    label=f"Next {horizon} Day(s) Horizon",
                                    value=f"{res['probability'] * 100:.1f}%",
                                    delta="High Risk" if res["prediction"] == 1 else "Normal Activity",
                                )
                                st.progress(res["probability"])
                                if res["prediction"] == 1:
                                    st.error(f"Unrest event expected (Threshold: {res['threshold'] * 100:.0f}%)")
                                else:
                                    st.success(f"Below threshold (Threshold: {res['threshold'] * 100:.0f}%)")

                    st.divider()
                    st.subheader("Ground Truth Validation")
                    st.caption("Observed outcomes recorded in the historical dataset following this date.")

                    comparison = []
                    for horizon in [1, 3, 7]:
                        predicted = results[horizon]["prediction"]
                        actual = int(row[f"target_{horizon}day"])
                        comparison.append({
                            "Horizon": f"{horizon} Day(s)",
                            "Predicted Probability": f"{results[horizon]['probability'] * 100:.1f}%",
                            "Predicted Decision": "Event Expected" if predicted == 1 else "No Event",
                            "Actual Outcome": "Event Occurred" if actual == 1 else "No Event",
                            "Verification": "Match" if predicted == actual else "Mismatch",
                        })

                    comparison_df = pd.DataFrame(comparison)
                    st.dataframe(comparison_df, hide_index=True)

                except Exception as e:
                    st.error(f"Forecast generation encountered an error: {e}")

        else:
            st.info("No recorded activity available for this state and date combination.")


# ═══════════════════════════════════════════════════════════
# TAB 4 — Integration Case Study
# ═══════════════════════════════════════════════════════════
with tab4:
    st.header("⚠️ Limited Case Study — India Farmers' Protest Only")
    st.markdown(
        "> **These are different tasks and cannot be directly compared. See RESULTS.md for details.**"
    )

    case_data = load_case_study()

    if case_data is None:
        st.info(
            "Case study not yet run. The file `results/case_study_results.json` does not exist. "
            "Run the case study pipeline to populate this tab.",
            icon=":material/schedule:",
        )
    else:
        with st.container(border=True):
            st.subheader("Case study summary")
            if "description" in case_data:
                st.markdown(case_data["description"])
            if "limitation" in case_data:
                st.warning(case_data["limitation"], icon="⚠️")
            info_cols = st.columns(3)
            if "data_coverage" in case_data:
                info_cols[0].metric("Data coverage", case_data["data_coverage"])
            if "model_used" in case_data:
                info_cols[1].metric("Model used", case_data["model_used"])

        # Metrics table from case study
        if "metrics" in case_data:
            st.subheader("Case study metrics")
            metrics_data = case_data["metrics"]
            if isinstance(metrics_data, dict):
                metrics_df = pd.DataFrame([metrics_data])
            elif isinstance(metrics_data, list):
                metrics_df = pd.DataFrame(metrics_data)
            else:
                metrics_df = None

            if metrics_df is not None:
                st.dataframe(metrics_df, hide_index=True)

    # Side-by-side comparison with CCC baseline
    st.subheader("Comparison: Case study vs CCC Baseline")
    final_df = load_final_results()
    if final_df is not None:
        st.caption("CCC Baseline (XGBoost, CCC 2021–2024 corpus)")
        st.dataframe(final_df, hide_index=True)
    else:
        st.info("`results/final_results.csv` not found. CCC baseline metrics unavailable.")

    st.markdown(
        "**These are different tasks and cannot be directly compared. See RESULTS.md for details.**"
    )


# ═══════════════════════════════════════════════════════════
# TAB 5 — Model Evaluation & Metrics
# ═══════════════════════════════════════════════════════════
with tab5:
    st.header("Model Evaluation & Metrics")
    st.caption(
        "Models trained on CCC data (2021–2022), validated on 2023, tested on 2024."
    )

    final_df = load_final_results()
    if final_df is not None:
        st.subheader("Performance across forecasting horizons")
        st.dataframe(final_df, hide_index=True)
    else:
        st.warning("`results/final_results.csv` not found.")

    # Thresholds and prevalence
    st.subheader("Operating thresholds & class prevalence")
    threshold_data = pd.DataFrame([
        {"Horizon": "1-day", "Threshold": 0.35, "Class prevalence (%)": 58.87},
        {"Horizon": "3-day", "Threshold": 0.25, "Class prevalence (%)": 77.94},
        {"Horizon": "7-day", "Threshold": 0.10, "Class prevalence (%)": 90.12},
    ])
    st.dataframe(threshold_data, hide_index=True)

    # Confusion matrices
    st.subheader("Confusion matrices (test set, 2024)")

    CONFUSION_MATRICES = {
        "1-day": {"matrix": [[5735, 2238], [1517, 9896]], "threshold": 0.35},
        "3-day": {"matrix": [[2383, 1894], [748, 14361]], "threshold": 0.25},
        "7-day": {"matrix": [[1011, 904], [231, 17240]], "threshold": 0.10},
    }

    cm_cols = st.columns(3)
    for col, (horizon, info) in zip(cm_cols, CONFUSION_MATRICES.items()):
        mat = info["matrix"]
        tn, fp, fn, tp = mat[0][0], mat[0][1], mat[1][0], mat[1][1]
        total = tn + fp + fn + tp
        with col:
            with st.container(border=True):
                st.markdown(f"**{horizon} horizon** (threshold: {info['threshold']})")
                cm_df = pd.DataFrame(
                    [
                        {"": "Predicted Neg", "Actual Neg": tn, "Actual Pos": fn},
                        {"": "Predicted Pos", "Actual Neg": fp, "Actual Pos": tp},
                    ]
                ).set_index("")
                st.dataframe(cm_df, alt=f"Confusion matrix for {horizon} horizon")
                st.caption(
                    f"TN={tn:,}  FP={fp:,}  FN={fn:,}  TP={tp:,}  "
                    f"(N={total:,})"
                )

    st.info(
        "Models trained on CCC data (2021–2022), validated on 2023, tested on 2024.",
        icon=":material/info:",
    )


# ---------------- FOOTER ----------------
st.divider()
st.caption(
    "Research Prototype | Predictions are probabilistic estimates based on state-level temporal patterns. "
    "Designed for academic benchmarking and evaluation; not intended for operational decision-making."
)
