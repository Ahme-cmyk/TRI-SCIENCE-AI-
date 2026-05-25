import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from PIL import Image
import os
import gdown
import keras

# Patch Keras
original_dense_init = keras.layers.Dense.__init__
def patched_dense_init(self, *args, **kwargs):
    kwargs.pop('quantization_config', None)
    original_dense_init(self, *args, **kwargs)
keras.layers.Dense.__init__ = patched_dense_init

# إعداد الواجهة
st.set_page_config(page_title="P.L.A.N.T. M.E.D. AI", layout="centered")
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Cairo:wght@400;700&display=swap');
    body { font-family: 'Cairo', sans-serif; direction: rtl; text-align: right; }
    .stApp { background-color: #f9f9f9; }
    </style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_all():
    file_id = '1dECn0z9QOpfcur4LH8XBTqQYQhCzV-i6'
    output_zip = 'ahmed_hosny.zip'
    if not os.path.exists(output_zip):
        gdown.download(id=file_id, output=output_zip, quiet=False)
    
    import zipfile
    with zipfile.ZipFile(output_zip, 'r') as zip_ref:
        zip_ref.extractall("models_folder")
        
    base = "models_folder"
    return (tf.keras.models.load_model(os.path.join(base, "Mint vs Basil.keras"), compile=False),
            tf.keras.models.load_model(os.path.join(base, "موديل السليم.keras"), compile=False),
            tf.keras.models.load_model(os.path.join(base, "موديل الامراض.keras"), compile=False),
            tf.keras.models.load_model(os.path.join(base, "disease_model_optimized.keras"), compile=False))

try:
    with st.spinner("جاري استدعاء العقول الأربعة..."):
        m_plant, m_health, m_disease, m_opt = load_all()
    st.success("✅ تم تفعيل العقول بنجاح!")
except Exception as e:
    st.error(f"❌ خطأ: {e}")
    st.stop()

def get_info(idx):
    # يمكنك تحديث هذا القاموس ليناسب الموديلات الجديدة
    return {0: "البياض الزغبي", 1: "تبقع الأوراق", 2: "البياض الدقيقي", 3: "صدأ الأوراق"}.get(idx, "مرض غير مصنف")

def predict(img_file):
    img = Image.open(img_file)
    st.image(img, use_container_width=True)
    
    x160 = np.expand_dims(image.img_to_array(img.resize((160, 160))), axis=0) / 255.0
    x224 = np.expand_dims(image.img_to_array(img.resize((224, 224))), axis=0) / 255.0
    
    p_type = np.argmax(m_plant.predict(x224, verbose=0))
    health_val = m_health.predict(x160, verbose=0)[0][0]
    
    st.write(f"### النوع: {'ريحان' if p_type == 0 else 'نعناع'}")
    
    if health_val < 0.3:
        st.success("✅ العينة سليمة")
    else:
        # هنا التشخيص المزدوج (موديل الأمراض + الموديل المحسن)
        dis_idx = np.argmax(m_disease.predict(x160, verbose=0))
        st.error(f"⚠️ العينة مصابة بـ: {get_info(dis_idx)}")

tab1, tab2 = st.tabs(["📸 كاميرا", "📁 ملفات"])
with tab1:
    photo = st.camera_input("التقط")
    if photo: predict(photo)
with tab2:
    up = st.file_uploader("ارفع", type=["jpg", "png"])
    if up: predict(up)
