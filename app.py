import streamlit as st
import pandas as pd
import numpy as np
import joblib

st.set_page_config(page_title="Nano-Drug Delivery Predictor", page_icon="🧬", layout="centered")
st.title("🧬 Nano-Particle Biodistribution Predictor")
st.markdown("Enter the nano-particle characteristics below to predict **Tumor Retention** and **Selectivity Index**.")
st.write("---")

# تحميل الموديل المحفوظ والمحدث
@st.cache_resource
def load_nano_model():
    return joblib.load('xgboost_nano_model.pkl')

try:
    model = load_nano_model()

    st.subheader("📊 Input Features")
    col1, col2 = st.columns(2)
    
    with col1:
        size = st.number_input("Size (nm):", min_value=1.0, max_value=1000.0, value=100.0)
        dosage = st.number_input("Administration Dosages (mg/kg):", min_value=0.0, max_value=500.0, value=5.0)
        time_point = st.number_input("Time point (h):", min_value=0.0, max_value=1000.0, value=24.0)
        zeta_mv = st.text_input("Zeta Potential (mv) [Leave empty if unknown]:", value="")
        
    with col2:
        # 1. تصحيح الـ NP Class بناء على الداتا الحقيقية
        np_class = st.selectbox("NP Class:", ["", "Organic", "Inorganic"])
        
        shape = st.selectbox("Shape:", ["", "Spherical", "Rod", "Cylinder", "Discoid", "Cubical"])
        zeta_cat = st.selectbox("Zeta Category:", ["", "Positive", "Negative", "Neutral"])
        has_peg = st.selectbox("HAS PEG:", ["", "Yes", "No"])
        
        # 2. تصحيح الـ Shell Type بناء على صورة الـ Excel
        shell_type = st.selectbox("Shell Type:", [
            "", "PEG", "Cellulose", "Dextran", "Fuc", "HA", "HPMA", "No Stealth Effect", "PKP"
        ])
        
        # 3. تصحيح الـ Tumor Site وإضافة كل الأنواع الناقصة
        tumor_site = st.selectbox("Tumor Site:", [
            "", "Cervix", "Brain", "Breast", "Colon", "Liver", "Lungs", 
            "Lymphoma", "Ovary", "Pancreas", "Prostate", "Sarcoma", "Skin"
        ])

    st.write("---")
    if st.button("🚀 Predict Target Outputs", type="primary"):
        # تجهيز البيانات للتنبؤ بنفس الأسماء الحرفية في الموديل
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
        
        # إجراء التنبؤ
        prediction = model.predict(input_df)
        
        st.subheader("🎯 Prediction Results")
        res_col1, res_col2 = st.columns(2)
        with res_col1:
            st.metric(label="📊 Predicted Tumor %ID", value=f"{prediction[0][0]:.4f} %")
        with res_col2:
            st.metric(label="🎯 Predicted Selectivity Index", value=f"{prediction[0][1]:.4f}")

except Exception as e:
    st.error(f"⚠️ حدث خطأ أثناء تشغيل الموديل: {e}")
    st.info("نصيحة: تأكدي من إعادة تشغيل سكريبت التدريب train_model.py إذا قمتِ بتغيير أعمدة الداتا.")
