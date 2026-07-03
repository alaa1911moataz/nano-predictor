import streamlit as st
import pandas as pd
import numpy as np
import joblib

# 1. إعدادات الصفحة وشكل الموقع
st.set_page_config(page_title="Nano-Drug Delivery Predictor", page_icon="🧬", layout="centered")

st.title("🧬 Nano-Particle Biodistribution Predictor")
st.markdown("Enter the nano-particle characteristics below to predict **Tumor Retention** and **Selectivity Index** using our trained XGBoost Model.")
st.write("---")

# 2. تحميل الموديل المحفوظ في الخلفية
@st.cache_resource # الحركة دي بتخلي الموقع سريع جداً لأنه بيحمل الموديل مرة واحدة بس
def load_nano_model():
    return joblib.load('xgboost_nano_model.pkl')

model = load_nano_model()

# 3. تصميم واجهة المستخدم (صناديق الإدخال والقوائم المنسدلة)
st.subheader("📊 Input Features")

col1, col2 = st.columns(2)

with col1:
    size = st.number_input("Size (nm):", min_value=1.0, max_value=500.0, value=100.0, step=1.0)
    dosage = st.number_input("Administration Dosages (mg/kg):", min_value=0.1, max_value=200.0, value=5.0, step=0.5)
    time_point = st.number_input("Time point (h):", min_value=0.0, max_value=500.0, value=24.0, step=1.0)
    zeta_mv = st.text_input("Zeta Potential (mv) [Leave empty if unknown]:", value="")

with col2:
    np_class = st.selectbox("NP Class:", ["", "Liposome", "Polymeric", "Silica", "Gold", "Magnetic"])
    shape = st.selectbox("Shape:", ["", "Spherical", "Rod", "Cylinder", "Discoid"])
    zeta_cat = st.selectbox("Zeta Category:", ["", "Positive", "Negative", "Neutral"])
    has_peg = st.selectbox("HAS PEG:", ["", "Yes", "No"])
    shell_type = st.selectbox("Shell Type:", ["", "Core-Shell", "Bare", "Functionalized"])
    tumor_site = st.selectbox("Tumor Site:", ["", "Breast", "Liver", "Lung", "Brain"])

st.write("---")

# 4. زرار التنبؤ وحساب النتائج
if st.button("🚀 Predict Target Outputs", type="primary"):
    
    # تحضير البيانات المستقبلة وتحويل الفراغات لـ np.nan تلقائياً لتغذية الـ XGBoost الفطري
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
    
    # تحويل لـ DataFrame
    input_df = pd.DataFrame([input_dict])
    
    # تشغيل الموديل
    with st.spinner("Calculating predictions..."):
        prediction = model.predict(input_df)
    
    # 5. عرض النتائج بشكل كروت جذابة (KPI Cards)
    st.subheader("🎯 Prediction Results")
    res_col1, res_col2 = st.columns(2)
    
    with res_col1:
        st.metric(label="📊 Predicted Tumor %ID", value=f"{prediction[0][0]:.4f} %")
        
    with res_col2:
        st.metric(label="🎯 Predicted Selectivity Index", value=f"{prediction[0][1]:.4f}")