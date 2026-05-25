import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from PIL import Image
import os
import zipfile
import keras

# Patch Keras
original_dense_init = keras.layers.Dense.__init__
def patched_dense_init(self, *args, **kwargs):
    kwargs.pop('quantization_config', None)
    original_dense_init(self, *args, **kwargs)
keras.layers.Dense.__init__ = patched_dense_init

st.set_page_config(page_title="P.L.A.N.T. M.E.D. AI", layout="centered")

# --- دالة فك الضغط والتحميل ---
@st.cache_resource
def prepare_models():
    zip_filename = "احمد حسني.zip"
    extract_folder = "extracted_models"
    
    # فك الضغط إذا لم يكن المجلد موجوداً
    if not os.path.exists(extract_folder):
        with zipfile.ZipFile(zip_filename, 'r') as zip_ref:
            zip_ref.extractall(extract_folder)
            
    # البحث عن الملفات داخل المجلد المستخرج (قد تكون داخل مجلد فرعي)
    models = {}
    for root, dirs, files in os.walk(extract_folder):
        for file in files:
            if file.endswith(".keras"):
                models[file] = os.path.join(root, file)
    return models

# تنفيذ التحميل
try:
    model_paths = prepare_models()
    model_plant = tf.keras.models.load_model(model_paths["Mint vs Basil.keras"], compile=False)
    model_disease = tf.keras.models.load_model(model_paths["موديل الامراض.keras"], compile=False)
    model_health = tf.keras.models.load_model(model_paths["موديل السليم.keras"], compile=False)
except Exception as e:
    st.error(f"خطأ في الوصول للموديلات داخل الملف المضغوط: {e}")
    st.stop()

# --- دالة التنبؤ ---
def process_and_predict(img_file):
    img = Image.open(img_file)
    st.image(img, use_container_width=True)
    
    x_224 = np.expand_dims(image.img_to_array(img.resize((224, 224))), axis=0) / 255.0
    x_160 = np.expand_dims(image.img_to_array(img.resize((160, 160))), axis=0) / 255.0

    plant_idx = np.argmax(model_plant.predict(x_224, verbose=0))
    raw_health = model_health.predict(x_160, verbose=0)[0][0]
    
    st.write(f"### نوع النبات: {'ريحان' if plant_idx == 0 else 'نعناع'}")
    
    if raw_health < 0.3:
        st.success("✅ العينة سليمة")
    else:
        dis_idx = np.argmax(model_disease.predict(x_160, verbose=0))
        st.error(f"⚠️ العينة مصابة بمرض رقم: {dis_idx}")

# الواجهة
tab1, tab2 = st.tabs(["📸 مسح ضوئي", "📁 رفع ملف"])
with tab1:
    photo = st.camera_input("التقط صورة")
    if photo: process_and_predict(photo)
with tab2:
    up = st.file_uploader("ارفع صورة", type=["jpg", "png"])
    if up: process_and_predict(up)
