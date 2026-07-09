import streamlit as st
import pandas as pd
import joblib
import numpy as np

# ==========================
# Page Configuration
# ==========================
st.set_page_config(
    page_title="Nano Delivery Predictor",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ==========================
# Load Trained Pipeline
# ==========================
@st.cache_resource
def load_model():
    model = joblib.load("xgboost_nano_classifier_model.pkl")
    return model

try:
    model = load_model()
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

# ==========================
# Feature Order (Must Match Training)
# ==========================
ordered_features = [
    "NP_Class",
    "INPs_Core",
    "Shape",
    "Size (nm)",
    "Size_Category",
    "Zeta Potential (mv)",
    "Zeta_Category",
    "Organ or tissue",
    "HAS_PEG",
    "Shell Type",
    "Administration Dosages (mg/kg)",
    "Time point (h)",
    "Tumor Site"
]

# ==========================
# Title
# ==========================
st.title("🧬 Nano Delivery Prediction System")
st.markdown(
    """
Predict **Tumor Retention** and **Targeting Selectivity**
using the fine-tuned XGBoost classifier.
"""
)


# ==========================
# Input Form
# ==========================

st.header("Nanoparticle Information")

col1, col2 = st.columns(2)

with col1:

    np_class = st.selectbox(
        "NP Class",
        sorted([
            "Organic",
            "Inorganic",
            "Hybrid"
        ])
    )

    inps_core = st.text_input("INPs Core")

    shape = st.selectbox(
        "Shape",
        sorted([
            "Sphere",
            "Rod",
            "Cube",
            "Disk",
            "Other"
        ])
    )

    size_nm = st.number_input(
        "Size (nm)",
        min_value=0.0,
        value=100.0
    )

    size_category = st.selectbox(
        "Size Category",
        [
            "Small",
            "Medium",
            "Large"
        ]
    )

    zeta = st.number_input(
        "Zeta Potential (mv)",
        value=0.0
    )

    zeta_category = st.selectbox(
        "Zeta Category",
        [
            "Negative",
            "Neutral",
            "Positive"
        ]
    )


with col2:

    organ = st.text_input("Organ or tissue")

    has_peg = st.selectbox(
        "Has PEG",
        [
            "Yes",
            "No"
        ]
    )

    shell = st.text_input("Shell Type")

    dosage = st.number_input(
        "Administration Dosages (mg/kg)",
        min_value=0.0,
        value=1.0
    )

    time_point = st.number_input(
        "Time point (h)",
        min_value=0.0,
        value=24.0
    )

    tumor_site = st.text_input("Tumor Site")


# ==========================
# Build Input Record
# ==========================

input_dict = {
    "NP_Class": np_class,
    "INPs_Core": inps_core,
    "Shape": shape,
    "Size (nm)": size_nm,
    "Size_Category": size_category,
    "Zeta Potential (mv)": zeta,
    "Zeta_Category": zeta_category,
    "Organ or tissue": organ,
    "HAS_PEG": has_peg,
    "Shell Type": shell,
    "Administration Dosages (mg/kg)": dosage,
    "Time point (h)": time_point,
    "Tumor Site": tumor_site
}


# ==========================
# Prediction
# ==========================

if st.button("Predict"):

    try:

        # Build DataFrame using the original feature order
        input_df = pd.DataFrame([input_dict])[ordered_features]

        # Predict directly using the saved Pipeline
        prediction = model.predict(input_df)

        tumor_prediction = prediction[0][0]
        selectivity_prediction = prediction[0][1]

        tumor_label = (
            "High Tumor Retention"
            if tumor_prediction == 1
            else "Low Tumor Retention"
        )

        selectivity_label = (
            "High Targeting Selectivity"
            if selectivity_prediction == 1
            else "Low Targeting Selectivity"
        )

        st.success("Prediction Completed Successfully!")

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Tumor Retention")
            st.metric(
                label="Prediction",
                value=tumor_label
            )

        with col2:
            st.subheader("Targeting Selectivity")
            st.metric(
                label="Prediction",
                value=selectivity_label
            )

        st.markdown("---")

        st.markdown("### Prediction Guide")

        st.info("""
**High Tumor Retention**

The nanoparticle is predicted to achieve relatively high accumulation inside the tumor.

**Low Tumor Retention**

The nanoparticle is predicted to have relatively low tumor accumulation.

**High Targeting Selectivity**

The nanoparticle is expected to preferentially accumulate in the tumor compared with other tissues.

**Low Targeting Selectivity**

The nanoparticle is expected to show lower tumor selectivity.
""")

    except Exception as e:
        st.error(f"Prediction failed: {e}")
