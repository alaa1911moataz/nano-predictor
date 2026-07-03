import streamlit as st
import pandas as pd
import numpy as np
import joblib

st.set_page_config(
    page_title="Nano-Drug Delivery Predictor",
    page_icon="🧬",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# ============================================================
# 🎨 Custom Dark Theme CSS
# ============================================================
st.markdown("""
<style>
    /* الخلفية العامة */
    .stApp {
        background: radial-gradient(circle at top left, #161A23 0%, #0B0D12 60%);
        color: #E6E6E6;
    }

    /* إخفاء الهيدر والفوتر الافتراضي */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}

    /* عنوان الصفحة */
    .hero-title {
        text-align: center;
        font-size: 2.4rem;
        font-weight: 800;
        background: linear-gradient(90deg, #7C4DFF, #00E0C6);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    .hero-subtitle {
        text-align: center;
        color: #9AA1B2;
        font-size: 1rem;
        margin-bottom: 1.8rem;
    }

    /* بطاقة الإدخال */
    .glass-card {
        background: rgba(255, 255, 255, 0.03);
        border: 1px solid rgba(124, 77, 255, 0.25);
        border-radius: 18px;
        padding: 1.6rem 1.6rem 0.4rem 1.6rem;
        margin-bottom: 1.5rem;
        box-shadow: 0 8px 24px rgba(0,0,0,0.35);
    }

    .section-label {
        font-size: 1.05rem;
        font-weight: 700;
        color: #C9B8FF;
        margin-bottom: 0.8rem;
        display: flex;
        align-items: center;
        gap: 0.4rem;
    }

    /* الـ inputs والـ selectbox */
    div[data-baseweb="select"] > div, .stNumberInput input, .stTextInput input {
        background-color: #1B1F2A !important;
        border: 1px solid #2C3140 !important;
        border-radius: 10px !important;
        color: #E6E6E6 !important;
    }

    label {
        color: #B8BFCF !important;
        font-weight: 500 !important;
    }

    /* زرار التنبؤ */
    div.stButton > button {
        width: 100%;
        background: linear-gradient(90deg, #7C4DFF, #5B2EFF);
        color: white;
        border: none;
        border-radius: 12px;
        padding: 0.8rem 0;
        font-size: 1.05rem;
        font-weight: 700;
        letter-spacing: 0.3px;
        transition: all 0.25s ease;
        box-shadow: 0 4px 18px rgba(124, 77, 255, 0.35);
    }
    div.stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 22px rgba(124, 77, 255, 0.55);
    }

    /* نتائج التنبؤ */
    div[data-testid="stMetric"] {
        background: rgba(124, 77, 255, 0.08);
        border: 1px solid rgba(124, 77, 255, 0.3);
        border-radius: 16px;
        padding: 1rem 0.5rem;
        text-align: center;
    }
    div[data-testid="stMetricValue"] {
        color: #00E0C6 !important;
        font-size: 1.6rem !important;
    }
    div[data-testid="stMetricLabel"] {
        color: #B8BFCF !important;
    }

    hr {
        border-color: rgba(255,255,255,0.08) !important;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================
# 🧬 Header
# ============================================================
st.markdown('<div class="hero-title">🧬 Nano-Particle Biodistribution Predictor</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="hero-subtitle">Enter the nano-particle characteristics below to predict '
    '<b>Tumor Retention</b> and <b>Selectivity Index</b></div>',
    unsafe_allow_html=True
)


@st.cache_resource
def load_nano_model():
    return joblib.load('xgboost_nano_model.pkl')


try:
    model = load_nano_model()

    # ---------------- Physical Properties ----------------
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-label">⚙️ Physical Properties</div>', unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        size = st.number_input("Size (nm):", min_value=1.0, max_value=1000.0, value=100.0)
        zeta_mv = st.text_input("Zeta Potential (mv) [optional]:", value="")
    with col2:
        shape = st.selectbox("Shape:", ["", "Spherical", "Rod", "Cylinder", "Discoid", "Cubical"])
        zeta_cat = st.selectbox("Zeta Category:", ["", "Positive", "Negative", "Neutral"])
    st.markdown('</div>', unsafe_allow_html=True)

    # ---------------- Composition ----------------
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-label">🧪 Composition & Coating</div>', unsafe_allow_html=True)
    col3, col4 = st.columns(2)
    with col3:
        np_class = st.selectbox("NP Class:", ["", "Organic", "Inorganic"])
        has_peg = st.selectbox("HAS PEG:", ["", "Yes", "No"])
    with col4:
        shell_type = st.selectbox("Shell Type:", [
            "", "PEG", "Cellulose", "Dextran", "Fuc", "HA", "HPMA", "No Stealth Effect", "PKP"
        ])
    st.markdown('</div>', unsafe_allow_html=True)

    # ---------------- Dosing & Target ----------------
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-label">💉 Dosing & Target</div>', unsafe_allow_html=True)
    col5, col6 = st.columns(2)
    with col5:
        dosage = st.number_input("Administration Dosage (mg/kg):", min_value=0.0, max_value=500.0, value=5.0)
        time_point = st.number_input("Time point (h):", min_value=0.0, max_value=1000.0, value=24.0)
    with col6:
        tumor_site = st.selectbox("Tumor Site:", [
            "", "Cervix", "Brain", "Breast", "Colon", "Liver", "Lungs",
            "Lymphoma", "Ovary", "Pancreas", "Prostate", "Sarcoma", "Skin"
        ])
    st.markdown('</div>', unsafe_allow_html=True)

    predict_clicked = st.button("🚀 Predict Target Outputs", type="primary")

    if predict_clicked:
        input_dict = {
            'NP_Class': np_class if np_class != "" else np.nan,
            'Shape': shape if shape != "" else np.nan,
            'Size (nm)': float(size),
            'Zeta Potential (mv)': float(zeta_mv) if zeta_mv.strip() != "" else np.nan,
            'Zeta_Category': zeta_cat if zeta_cat != "" else np.nan,
            'HAS_PEG': has_peg if has_peg != "" else np.nan,
            'Shell Type': shell_type if shell_type != "" else np.nan,
            'Administration Dosages (mg/kg)': float(dosage),
            'Time point (h)': float(time_point),
            'Tumor Site': tumor_site if tumor_site != "" else np.nan
        }

        input_df = pd.DataFrame([input_dict])
        prediction = model.predict(input_df)

        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown('<div class="section-label">🎯 Prediction Results</div>', unsafe_allow_html=True)
        res_col1, res_col2 = st.columns(2)
        with res_col1:
            st.metric(label="📊 Predicted Tumor %ID", value=f"{prediction[0][0]:.4f} %")
        with res_col2:
            st.metric(label="🎯 Predicted Selectivity Index", value=f"{prediction[0][1]:.4f}")
        st.markdown('</div>', unsafe_allow_html=True)

except Exception as e:
    st.error(f"⚠️ حدث خطأ أثناء تشغيل الموديل: {e}")
    st.info("نصيحة: تأكدي من إعادة تشغيل سكريبت التدريب train_model.py إذا قمتِ بتغيير أعمدة الداتا.")
