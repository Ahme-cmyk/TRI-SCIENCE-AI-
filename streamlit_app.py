import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from PIL import Image
import os
import gdown
import zipfile
import keras

# Patch Keras
original_dense_init = keras.layers.Dense.__init__
def patched_dense_init(self, *args, **kwargs):
    kwargs.pop('quantization_config', None)
    original_dense_init(self, *args, **kwargs)
keras.layers.Dense.__init__ = patched_dense_init

st.set_page_config(page_title="P.L.A.N.T. M.E.D. AI", layout="centered")

@st.cache_resource
def load_all_models_recursive():
    file_id = '1dECn0z9QOpfcur4LH8XBTqQYQhCzV-i6'
    output_zip = 'ahmed_hosny.zip'
    
    # تحميل الملف إذا لم يكن موجوداً
    if not os.path.exists(output_zip):
        gdown.download(id=file_id, output=output_zip, quiet=False)
    
    # فك الضغط في مجلد مؤقت
    extract_to = "final_extraction"
    if not os.path.exists(extract_to):
        with zipfile.ZipFile(output_zip, 'r') as zip_ref:
            zip_ref.extractall(extract_to)
            
    # البحث المتعمق (Recursive Search) عن الملفات أينما كانت
    found_paths = {}
    for root, dirs, files in os.walk(extract_to):
        for file in files:
            if file.endswith(".keras"):
                found_paths[file] = os.path.join(root, file)
                
    # التحقق من الملفات الأربعة
    required = ["Mint vs Basil.keras", "موديل السليم.keras", "موديل الامراض.keras", "disease_model_optimized.keras"]
    models = {}
    
    for req in required:
        if req in found_paths:
            models[req] = tf.keras.models.load_model(found_paths[req], compile=False)
        else:
            # رسالة خطأ تفصيلية تخبرك بالضبط أين بحثنا
            st.error(f"❌ لم يتم العثور على الموديل: {req}. الملفات التي وجدتها هي: {list(found_paths.keys())}")
            st.stop()
            
    return models["Mint vs Basil.keras"], models["موديل السليم.keras"], models["موديل الامراض.keras"], models["disease_model_optimized.keras"]

# تحميل الموديلات
try:
    with st.spinner("جاري استكشاف المجلدات وفك العقول..."):
        m_plant, m_health, m_disease, m_opt = load_all_models_recursive()
    st.success("✅ تم العثور على العقول الأربعة وتفعيلها!")
except Exception as e:
    st.error(f"❌ خطأ أثناء التحميل: {e}")
    st.stop()

# (بقية كود التنبؤ والواجهة كما هي)
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
        dis_idx = np.argmax(m_disease.predict(x160, verbose=0))
        st.error(f"⚠️ العينة مصابة (تشخيص: {dis_idx})")

tab1, tab2 = st.tabs(["📸 كاميرا", "📁 ملفات"])
with tab1:
    photo = st.camera_input("التقط")
    if photo: predict(photo)
with tab2:
    up = st.file_uploader("ارفع", type=["jpg", "png"])
    if up: predict(up)
