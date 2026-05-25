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

# إعداد الصفحة
st.set_page_config(page_title="P.L.A.N.T. M.E.D. AI", page_icon="🌿", layout="centered")

# التصميم (الرسالة الترحيبية)
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Cairo:wght@400;600;800&display=swap');
    html, body, [class*="css"] { font-family: 'Cairo', sans-serif; text-align: right; direction: rtl; }
    .welcome-card { background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%); color: white; padding: 30px; border-radius: 20px; box-shadow: 0 10px 20px rgba(0,0,0,0.1); text-align: center; margin-bottom: 25px; }
    </style>
    <div class="welcome-card">
        <div style="font-size: 32px; font-weight: 800;">🌿 منصة P.L.A.N.T. M.E.D. AI</div>
        <div style="font-size: 16px;">نظام التشخيص البيئي المعتمد على عقول الذكاء الاصطناعي التتابعية.</div>
    </div>
""", unsafe_allow_html=True)

@st.cache_resource
def load_models():
    file_id = '1dECn0z9QOpfcur4LH8XBTqQYQhCzV-i6'
    output_zip = 'ahmed_hosny.zip'
    if not os.path.exists(output_zip):
        gdown.download(id=file_id, output=output_zip, quiet=False)
    
    extract_to = "models_folder"
    if not os.path.exists(extract_to):
        with zipfile.ZipFile(output_zip, 'r') as zip_ref:
            zip_ref.extractall(extract_to)
            
    found_paths = {}
    for root, dirs, files in os.walk(extract_to):
        for file in files:
            if file.endswith(".keras"):
                found_paths[file] = os.path.join(root, file)
                
    # تحميل الموديلات
    m1 = tf.keras.models.load_model(found_paths["Mint vs Basil.keras"], compile=False)
    m2 = tf.keras.models.load_model(found_paths["موديل السليم.keras"], compile=False)
    m3 = tf.keras.models.load_model(found_paths["موديل الامراض.keras"], compile=False)
    m4 = tf.keras.models.load_model(found_paths["disease_model_optimized.keras"], compile=False)
    return m1, m2, m3, m4

# تحميل الموديلات
try:
    m_plant, m_health, m_disease, m_opt = load_models()
    st.success("✅ تم تفعيل العقول بنجاح!")
except Exception as e:
    st.error(f"❌ خطأ تحميل: {e}")
    st.stop()

def predict(img_file):
    img = Image.open(img_file)
    st.image(img, use_container_width=True)
    
    # تحضير الصور
    x160 = np.expand_dims(image.img_to_array(img.resize((160, 160))), axis=0) / 255.0
    
    # الحصول على تنبؤات الموديلات
    health_val = m_health.predict(x160, verbose=0)[0][0]
    
    # عرض الأرقام الخام بوضوح تام
    st.write("---")
    st.write(f"### 🧪 تحليل العقل (قيم خام):")
    st.write(f"قيمة الصحة (0=سليم, 1=مصاب): **{health_val:.4f}**")
    
    # هنا نضع "عتبة الاختبار"
    # غير الرقم 0.5 هذا وجربه 0.1 أو 0.8
    THRESHOLD = 0.5 
    
    if health_val < THRESHOLD:
        st.success(f"النتيجة حسب الموديل: سليم (لأن القيمة {health_val:.4f} < {THRESHOLD})")
    else:
        dis_idx = np.argmax(m_disease.predict(x160, verbose=0))
        st.error(f"النتيجة حسب الموديل: مصاب (لأن القيمة {health_val:.4f} > {THRESHOLD})")
        st.write(f"التشخيص: {get_info(dis_idx)}")

tab1, tab2 = st.tabs(["📸 كاميرا", "📁 ملفات"])
with tab1:
    photo = st.camera_input("التقط")
    if photo: predict(photo)
with tab2:
    up = st.file_uploader("ارفع", type=["jpg", "png"])
    if up: predict(up)
