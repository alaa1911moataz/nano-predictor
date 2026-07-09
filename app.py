import streamlit as st
import pandas as pd
import numpy as np
import pickle

# 1. إعدادات الصفحة الأساسية
st.set_page_config(
    page_title="Nano-toxicity Predictor",
    page_icon="🧬",
    layout="centered"
)

# 2. تصميم الـ CSS المخصص (بدون كسر الـ HTML)
st.markdown("""
<style>
    /* تحسين شكل الحاويات الافتراضية لتبدو مثل Glassmorphism */
    div[data-testid="stVerticalBlockBorderWrapper"] {
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(10px);
        border-radius: 15px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        padding: 20px;
        margin-bottom: 25px;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.2);
    }
    
    /* تصميم العناوين الفرعية */
    .section-label {
        font-size: 1.2rem;
        font-weight: bold;
        color: #00ffcc;
        margin-bottom: 15px;
        display: flex;
        align-items: center;
        gap: 10px;
    }
    .section-icon {
        font-size: 1.4rem;
    }
    
    /* زر التوقع */
    .stButton>button {
        width: 100%;
        background: linear-gradient(45deg, #00c6ff, #0072ff);
        color: white;
        border: none;
        padding: 12px;
        border-radius: 8px;
        font-size: 1.1rem;
        font-weight: bold;
        transition: transform 0.2s;
    }
    .stButton>button:hover {
        transform: scale(1.02);
        color: white;
    }
</style>
""", unsafe_allow_html=True)

# 3. تحميل الملفات الذكية (مع التخزين المؤقت لمنع البطء)
@st.cache_resource
def load_models():
    try:
        with open("nano_model.pkl", "rb") as f:
            model = pickle.load(f)
        with open("nano_preprocessor.pkl", "rb") as f:
            encoding_maps = pickle.load(f)
        return model, encoding_maps
    except FileNotFoundError as e:
        st.error(f"❌ لم يتم العثور على ملفات النموذج: {e.filename}. تأكد من وجودها في نفس المجلد.")
        return None, None

model, encoding_maps = load_models()

# عنوان التطبيق
st.title("🧬 Nano-toxicity Prediction App")
st.write("قم بإدخال مواصفات الجزيئات النانوية للتنبؤ بنسبة الحيوية الخلوية (Cell Viability).")
st.write("---")

if model and encoding_maps:
    # -------------------------------------------------------------------------
    # Card 1: Physical Properties
    # -------------------------------------------------------------------------
    st.markdown('<div class="section-label"><span class="section-icon">⚙️</span> Physical Properties</div>', unsafe_allow_html=True)
    with st.container(border=True):
        col1, col2 = st.columns(2)
        with col1:
            size = st.number_input("Size (nm)", min_value=1.0, max_value=1000.0, value=100.0, step=1.0)
            zeta_mv = st.text_input("Zeta Potential (mv) — optional", value="")
        with col2:
            shape = st.selectbox("Shape", ["", "Spherical", "Rod", "Cylinder", "Discoid", "Cubical"])
            zeta_cat = st.selectbox("Zeta Category", ["", "Positive", "Negative", "Neutral"])

    # -------------------------------------------------------------------------
    # Card 2: Chemical & Material Properties
    # -------------------------------------------------------------------------
    st.markdown('<div class="section-label"><span class="section-icon">🧪</span> Chemical & Material Properties</div>', unsafe_allow_html=True)
    with st.container(border=True):
        col3, col4 = st.columns(2)
        with col3:
            np_class = st.selectbox("NP Class", ["", "Metal", "Metal Oxide", "Carbon-based", "Polymer", "Lipid"])
            core_mat = st.text_input("Core Material", value="")
        with col4:
            surf_chg = st.selectbox("Surface Charge", ["", "Positive", "Negative", "Neutral"])
            surf_coat = st.text_input("Surface Coating — optional", value="")

    # -------------------------------------------------------------------------
    # Card 3: Biological & Experimental Conditions
    # -------------------------------------------------------------------------
    st.markdown('<div class="section-label"><span class="section-icon">🧫</span> Biological & Experimental Conditions</div>', unsafe_allow_html=True)
    with st.container(border=True):
        col5, col6 = st.columns(2)
        with col5:
            cell_line = st.text_input("Cell Line", value="")
            cell_org = st.selectbox("Cell Organism", ["", "Human", "Mouse", "Rat"])
            cell_type = st.selectbox("Cell Type", ["", "Cancerous", "Normal"])
        with col6:
            exposure_dose = st.number_input("Exposure Dose (µg/mL)", min_value=0.0, max_value=5000.0, value=10.0, step=0.5)
            exposure_time = st.number_input("Exposure Time (hours)", min_value=1.0, max_value=168.0, value=24.0, step=1.0)
            assay = st.selectbox("Assay Type", ["", "MTT", "CCK-8", "LDH", "Alamar Blue"])

    # -------------------------------------------------------------------------
    # زر التوقع ومعالجة البيانات
    # -------------------------------------------------------------------------
    if st.button("Predict Cell Viability"):
        
        # تجميع البيانات في قاموس بنفس أسماء الأعمدة الأصلية بدقة
        input_dict = {
            'NP_Class': str(np_class).strip() if np_class != "" else "nan",
            'Core_Material': str(core_mat).strip() if core_mat != "" else "nan",
            'Size_nm': float(size),
            'Shape': str(shape).strip() if shape != "" else "nan",
            'Surface_Charge': str(surf_chg).strip() if surf_chg != "" else "nan",
            'Surface_Coating': str(surf_coat).strip() if surf_coat != "" else "nan",
            'Zeta_Potential_mV': float(zeta_mv) if zeta_mv.strip() != "" else np.nan,
            'Zeta_Category': str(zeta_cat).strip() if zeta_cat != "" else "nan",
            'Cell_Line': str(cell_line).strip() if cell_line != "" else "nan",
            'Cell_Organism': str(cell_org).strip() if cell_org != "" else "nan",
            'Cell_Type': str(cell_type).strip() if cell_type != "" else "nan",
            'Exposure_Dose_ug_mL': float(exposure_dose),
            'Exposure_Time_hours': float(exposure_time),
            'Assay': str(assay).strip() if assay != "" else "nan"
        }
        
        # تحويل القاموس إلى DataFrame (سطر واحد)
        df_input = pd.DataFrame([input_dict])
        
        # تطبيق الـ Encoding بناءً على القاموس الجاهز
        encoded_dict = {}
        for col in df_input.columns:
            val = df_input[col].iloc[0]
            if col in encoding_maps:
                # إذا كانت الميزة فئوية، ابحث عنها في الخريطة، وإذا لم تجدها ضع NaN
                encoded_dict[col] = encoding_maps[col].get(val, np.nan)
            else:
                # الميزات الرقمية تظل كما هي
                encoded_dict[col] = val
        
        # تحويل البيانات المشفرة إلى DataFrame نهائي بنفس ترتيب ميزات النموذج
        df_encoded = pd.DataFrame([encoded_dict])
        
        # محاكاة الترتيب لضمان مطابقة نموذج الـ XGBoost تماماً
        if hasattr(model, "feature_names_in_"):
            df_encoded = df_encoded.reindex(columns=model.feature_names_in_)
            
        try:
            # إجراء عملية التنبؤ
            prediction = model.predict(df_encoded)[0]
            
            # عرض النتيجة بشكل جمالي
            st.success("🎉 تم حساب التوقع بنجاح!")
            st.metric(label="Predicted Cell Viability (%)", value=f"{prediction:.2f}%")
            
            # شريط ملون توضيحي للنتيجة
            if prediction >= 70:
                st.info("🟢 تصنيف: جزيئات نانوية منخفضة السمية (آمنة نسبياً).")
            elif 40 <= prediction < 70:
                st.warning("🟡 تصنيف: سمية متوسطة.")
            else:
                st.error("🔴 تصنيف: سمية عالية جداً!")
                
        except Exception as error:
            st.error(f"حدث خطأ أثناء التنبؤ: {error}")
else:
    st.info("💡 يرجى رفع ملفات `nano_model.pkl` و `nano_preprocessor.pkl` في مجلد المشروع لتفعيل الواجهة.")
