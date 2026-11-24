import streamlit as st
import pandas as pd
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


# ---------------- LOAD DATA WITH ERROR BOUNDARY ----------------
@st.cache_data(show_spinner="Loading historical event database...")
def get_historical_data():
    try:
        return load_and_process_data()
    except FileNotFoundError:
        return None
    except Exception as e:
        st.error(f"Failed to load dataset: {e}")
        return None


data = get_historical_data()


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

if data is None:
    st.warning(
        "Historical CCC dataset (`data/ccc_compiled_20212024.csv`) was not found. "
        "Place the compiled dataset in `data/` to enable interactive historical lookups. "
        "The model inference service remains operational via `backend.predict`."
    )
    st.stop()


# ---------------- STATE AND DATE SELECTION ----------------
with st.container(border=True):
    st.subheader("Select Historical Region and Observation Date")
    col1, col2 = st.columns(2)

    available_states = sorted(data["state"].unique())

    with col1:
        selected_state = st.selectbox(
            "Select State / Region",
            available_states,
            index=available_states.index("CA") if "CA" in available_states else 0
        )

    state_data = data[data["state"] == selected_state].copy()
    min_date = state_data["date"].min().date()
    max_date = state_data["date"].max().date()

    with col2:
        selected_date = st.date_input(
            "Select Observation Date",
            value=max_date,
            min_value=min_date,
            max_value=max_date
        )

# Retrieve selected state and date record
selected_row = state_data[state_data["date"].dt.date == selected_date]


# ---------------- FEATURE EXTRACTION & FORECASTING ----------------
if not selected_row.empty:
    row = selected_row.iloc[0]
    features = {feat: float(row[feat]) for feat in FEATURES}

    st.subheader("Historical Activity Features")
    st.caption("Features extracted from historical event time-series up to observation date.")

    feature_cols = st.columns(4)
    feature_display = [
        ("Events yesterday", "lag_1"),
        ("Events 3 days ago", "lag_3"),
        ("Events 7 days ago", "lag_7"),
        ("Events 14 days ago", "lag_14"),
        ("7-day rolling average", "rolling_7_mean"),
        ("7-day total events", "rolling_7_sum"),
        ("14-day rolling average", "rolling_14_mean")
    ]

    for i, (label, key) in enumerate(feature_display):
        with feature_cols[i % 4]:
            with st.container(border=True):
                st.metric(label, f"{features[key]:.2f}")

    # Forecast generation trigger
    st.write("")
    if st.button("Generate Multi-Horizon Forecast", type="primary", width="stretch"):
        try:
            results = {
                1: predict_unrest(features, 1),
                3: predict_unrest(features, 3),
                7: predict_unrest(features, 7)
            }

            st.divider()
            st.subheader("Forecast Results")
            result_cols = st.columns(3)

            for col, horizon in zip(result_cols, [1, 3, 7]):
                res = results[horizon]
                with col:
                    with st.container(border=True):
                        st.metric(
                            label=f"Next {horizon} Day(s) Horizon",
                            value=f"{res['probability'] * 100:.1f}%",
                            delta="High Risk" if res["prediction"] == 1 else "Normal Activity"
                        )
                        st.progress(res["probability"])
                        if res["prediction"] == 1:
                            st.error(f"Unrest event expected (Threshold: {res['threshold'] * 100:.0f}%)")
                        else:
                            st.success(f"Below threshold (Threshold: {res['threshold'] * 100:.0f}%)")

            # Actual Outcomes Comparison
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
                    "Verification": "Match" if predicted == actual else "Mismatch"
                })

            comparison_df = pd.DataFrame(comparison)
            st.dataframe(comparison_df, hide_index=True)

        except Exception as e:
            st.error(f"Forecast generation encountered an error: {e}")

else:
    st.info("No recorded activity available for this state and date combination.")


# ---------------- FOOTER ----------------
st.divider()
st.caption(
    "Research Prototype | Predictions are probabilistic estimates based on state-level temporal patterns. "
    "Designed for academic benchmarking and evaluation; not intended for operational decision-making."
)
