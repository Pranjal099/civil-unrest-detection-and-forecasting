import streamlit as st
import pandas as pd

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
    background-color: #F4F7FB;
    color: #111827;
}

.block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
}

h1, h2, h3 {
    color: #111827 !important;
}

[data-testid="stSidebar"] {
    background-color: #1E293B;
}

[data-testid="stSidebar"] * {
    color: #FFFFFF !important;
}

[data-testid="stWidgetLabel"] p {
    color: #374151 !important;
    font-weight: 500;
}

[data-testid="stMetric"] {
    background-color: #FFFFFF;
    border: 1px solid #E5E7EB;
    padding: 18px;
    border-radius: 12px;
}

div.stButton > button {
    background-color: #2563EB;
    color: white;
    border: none;
    border-radius: 8px;
    height: 45px;
    font-weight: 600;
}

div.stButton > button:hover {
    background-color: #1D4ED8;
    color: white;
}

.subtitle {
    color: #64748B;
    font-size: 15px;
}

hr {
    border-color: #D1D5DB;
}
</style>
""", unsafe_allow_html=True)


# ---------------- LOAD DATA ----------------
@st.cache_data
def get_data():
    return load_and_process_data()


data = get_data()


# ---------------- SIDEBAR ----------------
with st.sidebar:

    st.markdown("## Civil Unrest")
    st.markdown("### Forecasting System")

    st.divider()

    st.markdown("**Model Information**")
    st.write("Algorithm: XGBoost")
    st.write("Forecast horizons: 1, 3, 7 days")
    st.write("Dataset: CCC 2021–2024")

    st.divider()

    st.caption("Academic ML project")
    st.caption("Historical activity forecasting")


# ---------------- HEADER ----------------
st.title("Civil Unrest Forecast Dashboard")

st.markdown(
    '<p class="subtitle">'
    'Historical activity-based forecasting using machine learning.'
    '</p>',
    unsafe_allow_html=True
)

st.divider()


# ---------------- STATE AND DATE SELECTION ----------------
st.subheader("Select Historical Data")

col1, col2 = st.columns(2)

available_states = sorted(data["state"].unique())

with col1:
    selected_state = st.selectbox(
        "Select State",
        available_states
    )

state_data = data[data["state"] == selected_state].copy()

min_date = state_data["date"].min().date()
max_date = state_data["date"].max().date()

with col2:
    selected_date = st.date_input(
        "Select Date",
        value=max_date,
        min_value=min_date,
        max_value=max_date
    )

# Retrieve selected state and date
selected_row = state_data[
    state_data["date"].dt.date == selected_date
]


# ---------------- FEATURE EXTRACTION ----------------
if not selected_row.empty:

    row = selected_row.iloc[0]

    features = {
        feature: float(row[feature])
        for feature in FEATURES
    }

    st.divider()

    st.subheader("Historical Activity Features")

    st.caption(
        "Features automatically extracted from the CCC dataset."
    )

    feature_cols = st.columns(4)

    feature_display = [
        ("Events yesterday", "lag_1"),
        ("Events 3 days ago", "lag_3"),
        ("Events 7 days ago", "lag_7"),
        ("Events 14 days ago", "lag_14"),
        ("7-day average", "rolling_7_mean"),
        ("7-day total", "rolling_7_sum"),
        ("14-day average", "rolling_14_mean")
    ]

    for i, (label, key) in enumerate(feature_display):

        with feature_cols[i % 4]:
            st.metric(
                label,
                f"{features[key]:.2f}"
            )


    # ---------------- PREDICTION ----------------
    st.divider()

    if st.button("Generate Forecast", use_container_width=True):

        try:

            results = {
                1: predict_unrest(features, 1),
                3: predict_unrest(features, 3),
                7: predict_unrest(features, 7)
            }

            st.subheader("Forecast Results")

            result_cols = st.columns(3)

            for col, horizon in zip(result_cols, [1, 3, 7]):

                result = results[horizon]

                with col:

                    st.metric(
                        label=f"Next {horizon} Day(s)",
                        value=f"{result['probability'] * 100:.2f}%"
                    )

                    st.progress(result["probability"])

                    if result["prediction"] == 1:
                        st.success("Positive forecast signal")
                    else:
                        st.warning("Below forecast threshold")

                    st.caption(
                        f"Threshold: "
                        f"{result['threshold'] * 100:.0f}%"
                    )


            # ---------------- ACTUAL OUTCOMES ----------------
            st.divider()

            st.subheader("Actual Outcomes")

            st.caption(
                "Actual event occurrence from the historical dataset."
            )

            actual_cols = st.columns(3)

            for col, horizon in zip(actual_cols, [1, 3, 7]):

                actual = int(row[f"target_{horizon}day"])

                with col:

                    st.metric(
                        f"Actual — {horizon} Day(s)",
                        "Event occurred" if actual == 1 else "No event"
                    )


            # ---------------- COMPARISON ----------------
            st.divider()

            st.subheader("Prediction vs Actual")

            comparison = []

            for horizon in [1, 3, 7]:

                predicted = results[horizon]["prediction"]
                actual = int(row[f"target_{horizon}day"])

                comparison.append({
                    "Forecast": f"{horizon} day(s)",
                    "Predicted": (
                        "Event expected"
                        if predicted == 1
                        else "No event expected"
                    ),
                    "Actual": (
                        "Event occurred"
                        if actual == 1
                        else "No event"
                    ),
                    "Result": (
                        "Correct"
                        if predicted == actual
                        else "Incorrect"
                    )
                })

            comparison_df = pd.DataFrame(comparison)

            st.dataframe(
                comparison_df,
                use_container_width=True,
                hide_index=True
            )

        except Exception as e:

            st.error(f"Prediction error: {e}")

else:

    st.warning("No data available for this state and date.")


# ---------------- FOOTER ----------------
st.divider()

st.caption(
    "Research prototype | Predictions are based on historical event-count "
    "features. Probabilities are model estimates, not certainties or causal "
    "explanations. This interface is not validated for operational decision-making."
)