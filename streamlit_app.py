import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from PIL import Image
import os
import keras

# 1. Patch Keras لتجنب أخطاء Serialization
original_dense_init = keras.layers.Dense.__init__
def patched_dense_init(self, *args, **kwargs):
    kwargs.pop('quantization_config', None)
    original_dense_init(self, *args, **kwargs)
keras.layers.Dense.__init__ = patched_dense_init

st.set_page_config(page_title="P.L.A.N.T. M.E.D. AI", layout="centered")

# 2. تحميل الموديلات مع فحص المسارات
@st.cache_resource
def load_models():
    base_path = "models_data"
    # مسارات الملفات
    p_plant = os.path.join(base_path, "Mint vs Basil.keras")
    p_disease = os.path.join(base_path, "موديل الامراض.keras")
    p_health = os.path.join(base_path, "موديل السليم.keras")
    
    # فحص الوجود
    for p in [p_plant, p_disease, p_health]:
        if not os.path.exists(p):
            raise FileNotFoundError(f"الملف غير موجود: {p}. تأكد من رفع المجلد إلى GitHub.")
            
    # التحميل
    return (tf.keras.models.load_model(p_plant, compile=False), 
            tf.keras.models.load_model(p_disease, compile=False), 
            tf.keras.models.load_model(p_health, compile=False))

# تحميل الموديلات
try:
    model_plant, model_disease, model_health = load_models()
except Exception as e:
    st.error(f"خطأ في تحميل الموديلات: {e}")
    st.stop()

# 3. دالة التشخيص
def process_and_predict(img_file):
    img = Image.open(img_file)
    st.image(img, use_container_width=True)
    
    # تحضير الصور
    x_224 = np.expand_dims(image.img_to_array(img.resize((224, 224))), axis=0) / 255.0
    x_160 = np.expand_dims(image.img_to_array(img.resize((160, 160))), axis=0) / 255.0

    # التنبؤ
    plant_idx = np.argmax(model_plant.predict(x_224, verbose=0))
    raw_health = model_health.predict(x_160, verbose=0)[0][0]
    
    # عرض النتائج
    st.write(f"### نوع النبات: {'ريحان' if plant_idx == 0 else 'نعناع'}")
    
    # كواليس التصحيح
    with st.sidebar:
        st.write(f"الرقم الخام للصحة: {raw_health:.4f}")
    
    # قرار التشخيص (بناءً على 0.3 كعتبة حساسة)
    if raw_health < 0.3:
        st.success("✅ العينة سليمة")
    else:
        dis_idx = np.argmax(model_disease.predict(x_160, verbose=0))
        st.error(f"⚠️ العينة مصابة بمرض رقم: {dis_idx}")

# 4. الواجهة
tab1, tab2 = st.tabs(["📸 مسح ضوئي", "📁 رفع ملف"])
with tab1:
    photo = st.camera_input("التقط صورة")
    if photo: process_and_predict(photo)
with tab2:
    up = st.file_uploader("ارفع صورة", type=["jpg", "png"])
    if up: process_and_predict(up)
