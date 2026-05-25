import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from PIL import Image
import os
import zipfile
import urllib.request
import keras

# 1. Patch Keras
original_dense_init = keras.layers.Dense.__init__
def patched_dense_init(self, *args, **kwargs):
    kwargs.pop('quantization_config', None)
    original_dense_init(self, *args, **kwargs)
keras.layers.Dense.__init__ = patched_dense_init

# 2. دالة التحميل الذكي (المفقودة في كودك)
@st.cache_resource
def download_models():
    zip_path = "models.zip"
    extract_path = "models_data"
    if not os.path.exists(extract_path):
        url = "https://huggingface.co/ahmedhosny2052005/TRI-SCIENCE-AI/resolve/main/%D8%A7%D8%AD%D9%85%D8%AF%20%D8%AD%D8%B3%D9%86%D9%8A.zip"
        urllib.request.urlretrieve(url, zip_path)
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(".") # سيفك الضغط في مجلد اسمه models_data
    return extract_path

# 3. إعداد الصفحة والتحميل
st.set_page_config(page_title="P.L.A.N.T. M.E.D. AI", layout="centered")
download_models()

@st.cache_resource
def load_all_models():
    model_plant = tf.keras.models.load_model("models_data/Mint vs Basil.keras", compile=False)
    model_disease = tf.keras.models.load_model("models_data/موديل الامراض.keras", compile=False)
    model_health = tf.keras.models.load_model("models_data/موديل السليم.keras", compile=False)
    return model_plant, model_disease, model_health

model_plant, model_disease, model_health = load_all_models()

# 4. دالة المعالجة
def process_and_predict(image_data):
    img = Image.open(image_data)
    st.image(img, use_container_width=True)
    
    img_224 = img.resize((224, 224))
    x_224 = np.expand_dims(image.img_to_array(img_224), axis=0) / 255.0
    
    img_160 = img.resize((160, 160))
    x_160 = np.expand_dims(image.img_to_array(img_160), axis=0) / 255.0

    # التنبؤ
    plant_preds = model_plant.predict(x_224)
    detected_plant = "ريحان" if np.argmax(plant_preds) == 0 else "نعناع"
    
    health_preds = model_health.predict(x_160)
    raw_val = health_preds[0][0]
    
    # تصحيح العتبة (جرب 0.2 أو 0.3 إذا لم يكتشف المرض)
    is_healthy = raw_val < 0.3 
    
    st.write(f"### نوع النبات: {detected_plant}")
    if is_healthy:
        st.success("العينة سليمة")
    else:
        dis_preds = model_disease.predict(x_160)
        dis_idx = np.argmax(dis_preds)
        st.error(f"العينة مصابة بمرض رقم: {dis_idx}")

# 5. الواجهة
tab1, tab2 = st.tabs(["كاميرا", "ملفات"])
with tab1:
    photo = st.camera_input("التقط صورة")
    if photo: process_and_predict(photo)
with tab2:
    up = st.file_uploader("ارفع صورة", type=["jpg", "png"])
    if up: process_and_predict(up)
