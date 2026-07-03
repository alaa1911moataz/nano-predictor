import streamlit as st
import pandas as pd
import numpy as np
import joblib

st.set_page_config(
    page_title="Nano-Drug Delivery Predictor",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="collapsed",
)

if "theme" not in st.session_state:
    st.session_state.theme = "light"

def toggle_theme():
    st.session_state.theme = "dark" if st.session_state.theme == "light" else "light"

is_dark = st.session_state.theme == "dark"

if is_dark:
    bg = "#0A0C12"
    bg_gradient = """
        radial-gradient(circle at 15% 10%, rgba(124, 77, 255, 0.15) 0%, transparent 40%),
        radial-gradient(circle at 85% 90%, rgba(0, 224, 198, 0.12) 0%, transparent 40%),
        #0A0C12
    """
    text_color = "#E8E9ED"
    subtitle_color = "#8B92A6"
    card_bg = "linear-gradient(180deg, rgba(255,255,255,0.035), rgba(255,255,255,0.015))"
    card_border = "rgba(255,255,255,0.08)"
    card_border_hover = "rgba(124, 77, 255, 0.35)"
    input_bg = "#14161F"
    input_border = "#262A38"
    label_color = "#9AA1B4"
    section_title_color = "#FFFFFF"
    footer_color = "#575E70"
    shadow = "0 10px 30px rgba(0,0,0,0.4), inset 0 1px 0 rgba(255,255,255,0.04)"
else:
    bg = "#F7F7FB"
    bg_gradient = """
        radial-gradient(circle at 15% 10%, rgba(124, 77, 255, 0.08) 0%, transparent 45%),
        radial-gradient(circle at 85% 90%, rgba(0, 191, 165, 0.08) 0%, transparent 45%),
        #F7F7FB
    """
    text_color = "#1E2130"
    subtitle_color = "#6B7280"
    card_bg = "linear-gradient(180deg, rgba(255,255,255,0.95), rgba(255,255,255,0.85))"
    card_border = "rgba(124, 77, 255, 0.12)"
    card_border_hover = "rgba(124, 77, 255, 0.4)"
    input_bg = "#FFFFFF"
    input_border = "#E2E4EE"
    label_color = "#5B6072"
    section_title_color = "#1E2130"
    footer_color = "#9CA0AE"
    shadow = "0 10px 26px rgba(124, 77, 255, 0.10), inset 0 1px 0 rgba(255,255,255,0.6)"

st.markdown(f"""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700;800&display=swap');

    html, body, [class*="css"] {{
        font-family: 'Poppins', sans-serif;
    }}

    .stApp {{
        background: {bg_gradient};
        color: {text_color};
    }}

    #MainMenu {{visibility: hidden;}}
    footer {{visibility: hidden;}}
    header {{visibility: hidden;}}

    .block-container {{
        padding-top: 2rem;
        padding-left: 1rem;
        padding-right: 1rem;
        max-width: 1100px;
    }}

    div[data-testid="column"] {{
        padding: 0 0.4rem;
    }}

    div[data-baseweb="select"],
    div[data-baseweb="select"] > div,
    .stNumberInput,
    .stTextInput {{
        width: 100% !important;
    }}

    div[data-testid="stVerticalBlock"] > div {{
        gap: 0.6rem;
    }}

    .theme-toggle-row {{
        display: flex;
        justify-content: flex-end;
        margin-bottom: 0.5rem;
    }}

    .hero-wrap {{
        text-align: center;
        margin-bottom: 2.2rem;
    }}
    .hero-badge {{
        display: inline-block;
        padding: 0.35rem 1rem;
        border-radius: 999px;
        background: rgba(124, 77, 255, 0.10);
        border: 1px solid rgba(124, 77, 255, 0.30);
        color: #7C4DFF;
        font-size: 0.78rem;
        font-weight: 600;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        margin-bottom: 1rem;
    }}
    .hero-title {{
        font-size: 2.6rem;
        font-weight: 800;
        line-height: 1.15;
        background: linear-gradient(100deg, #7C4DFF 0%, #00BFA5 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.6rem;
    }}
    .hero-subtitle {{
        color: {subtitle_color};
        font-size: 1.02rem;
        font-weight: 400;
        max-width: 560px;
        margin: 0 auto;
        line-height: 1.6;
    }}

    .glass-card {{
        background: {card_bg};
        backdrop-filter: blur(12px);
        border: 1px solid {card_border};
        border-radius: 20px;
        padding: 1.7rem 1.8rem 0.5rem 1.8rem;
        margin-bottom: 1.4rem;
        box-shadow: {shadow};
        transition: border 0.25s ease;
    }}
    .glass-card:hover {{
        border: 1px solid {card_border_hover};
    }}

    .section-label {{
        font-size: 1rem;
        font-weight: 600;
        color: {section_title_color};
        margin-bottom: 1.1rem;
        display: flex;
        align-items: center;
        gap: 0.55rem;
        padding-bottom: 0.8rem;
        border-bottom: 1px solid {card_border};
    }}
    .section-icon {{
        width: 32px; height: 32px;
        border-radius: 9px;
        display: inline-flex;
        align-items: center;
        justify-content: center;
        background: linear-gradient(135deg, rgba(124,77,255,0.20), rgba(0,191,165,0.18));
        font-size: 1rem;
    }}

    div[data-baseweb="select"] > div, .stNumberInput input, .stTextInput input {{
        background-color: {input_bg} !important;
        border: 1px solid {input_border} !important;
        border-radius: 11px !important;
        color: {text_color} !important;
        font-size: 0.92rem !important;
    }}
    div[data-baseweb="select"] > div:focus-within, .stNumberInput input:focus, .stTextInput input:focus {{
        border: 1px solid #7C4DFF !important;
        box-shadow: 0 0 0 3px rgba(124, 77, 255, 0.15) !important;
    }}
    label {{
        color: {label_color} !important;
        font-weight: 500 !important;
        font-size: 0.86rem !important;
    }}

    div.stButton > button {{
        width: 100%;
        background: linear-gradient(100deg, #7C4DFF, #00BFA5);
        color: white;
        border: none;
        border-radius: 13px;
        padding: 0.85rem 0;
        font-size: 1.02rem;
        font-weight: 700;
        letter-spacing: 0.3px;
        transition: all 0.25s ease;
        box-shadow: 0 6px 20px rgba(124, 77, 255, 0.25);
    }}
    div.stButton > button:hover {{
        transform: translateY(-2px);
        box-shadow: 0 10px 28px rgba(124, 77, 255, 0.45);
    }}
    div.stButton > button:active {{
        transform: translateY(0px);
    }}

    .result-title {{
        text-align: center;
        font-size: 1.3rem;
        font-weight: 700;
        color: {section_title_color};
        margin: 1.6rem 0 1rem 0;
    }}
    div[data-testid="stMetric"] {{
        background: linear-gradient(180deg, rgba(0, 191, 165, 0.10), rgba(124, 77, 255, 0.06));
        border: 1px solid rgba(0, 191, 165, 0.30);
        border-radius: 18px;
        padding: 1.3rem 0.6rem;
        text-align: center;
        box-shadow: 0 8px 20px rgba(0, 191, 165, 0.08);
    }}
    div[data-testid="stMetricValue"] {{
        color: #00A896 !important;
        font-size: 1.9rem !important;
        font-weight: 700 !important;
    }}
    div[data-testid="stMetricLabel"] {{
        color: {label_color} !important;
        font-size: 0.85rem !important;
    }}

    hr {{ border-color: {card_border} !important; }}

    .footer-note {{
        text-align: center;
        color: {footer_color};
        font-size: 0.78rem;
        margin-top: 2rem;
        padding-bottom: 1rem;
    }}
</style>
""", unsafe_allow_html=True)

toggle_col1, toggle_col2 = st.columns([6, 1])
with toggle_col2:
    st.button("🌙 Dark" if not is_dark else "☀️ Light", on_click=toggle_theme, use_container_width=True)

st.markdown("""
<div class="hero-wrap">
    <div class="hero-badge">🧬 AI-Powered Biodistribution Model</div>
    <div class="hero-title">Nano-Particle Biodistribution<br>Predictor</div>
    <div class="hero-subtitle">
        Enter the nano-particle characteristics below to predict
        <b style="color:#7C4DFF;">Tumor Retention</b> and
        <b style="color:#00A896;">Selectivity Index</b> using a trained XGBoost model.
    </div>
</div>
""", unsafe_allow_html=True)


@st.cache_resource
def load_nano_model():
    return joblib.load('xgboost_nano_model.pkl')


try:
    model = load_nano_model()

    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-label"><span class="section-icon">⚙️</span> Physical Properties</div>', unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        size = st.number_input("Size (nm)", min_value=1.0, max_value=1000.0, value=100.0)
        zeta_mv = st.text_input("Zeta Potential (mv) — optional", value="")
    with col2:
        shape = st.selectbox("Shape", ["", "Spherical", "Rod", "Cylinder", "Discoid", "Cubical"])
        zeta_cat = st.selectbox("Zeta Category", ["", "Positive", "Negative", "Neutral"])
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-label"><span class="section-icon">🧪</span> Composition & Coating</div>', unsafe_allow_html=True)
    col3, col4 = st.columns(2)
    with col3:
        np_class = st.selectbox("NP Class", ["", "Organic", "Inorganic"])
        has_peg = st.selectbox("Has PEG", ["", "Yes", "No"])
    with col4:
        shell_type = st.selectbox("Shell Type", [
            "", "PEG", "Cellulose", "Dextran", "Fuc", "HA", "HPMA", "No Stealth Effect", "PKP"
        ])
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-label"><span class="section-icon">💉</span> Dosing & Target</div>', unsafe_allow_html=True)
    col5, col6 = st.columns(2)
    with col5:
        dosage = st.number_input("Administration Dosage (mg/kg)", min_value=0.0, max_value=500.0, value=5.0)
        time_point = st.number_input("Time Point (h)", min_value=0.0, max_value=1000.0, value=24.0)
    with col6:
        tumor_site = st.selectbox("Tumor Site", [
            "", "Cervix", "Brain", "Breast", "Colon", "Liver", "Lungs",
            "Lymphoma", "Ovary", "Pancreas", "Prostate", "Sarcoma", "Skin"
        ])
    st.markdown('</div>', unsafe_allow_html=True)

    predict_clicked = st.button("  Predict Target Outputs  ", type="primary")

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

        st.markdown('<div class="result-title">🎯 Prediction Results</div>', unsafe_allow_html=True)
        res_col1, res_col2 = st.columns(2)
        with res_col1:
            st.metric(label="📊 Predicted Tumor %ID", value=f"{prediction[0][0]:.4f} %")
        with res_col2:
            st.metric(label="🎯 Predicted Selectivity Index", value=f"{prediction[0][1]:.4f}")

    st.markdown('<div class="footer-note">Powered by XGBoost · Built with Streamlit</div>', unsafe_allow_html=True)

except Exception as e:
    st.error(f"⚠️ حدث خطأ أثناء تشغيل الموديل: {e}")
    st.info("نصيحة: تأكدي من إعادة تشغيل سكريبت التدريب train_model.py إذا قمتِ بتغيير أعمدة الداتا.")
