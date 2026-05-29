import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from PIL import Image
import os
import gdown
import zipfile
import keras

# --- 1. إصلاح توافقية كراس ---
original_dense_init = keras.layers.Dense.__init__
def patched_dense_init(self, *args, **kwargs):
    kwargs.pop('quantization_config', None)
    original_dense_init(self, *args, **kwargs)
keras.layers.Dense.__init__ = patched_dense_init

# --- 2. دالة تحميل الموديلات من جوجل درايف ---
@st.cache_resource
def load_models():
    file_id = '1dECn0z9QOpfcur4LH8XBTqQYQhCzV-i6'
    output_zip = 'موديل.zip'

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

    m1 = tf.keras.models.load_model(found_paths["Mint vs Basil.keras"], compile=False)
    m2 = tf.keras.models.load_model(found_paths["موديل السليم.keras"], compile=False)
    m3 = tf.keras.models.load_model(found_paths["موديل الامراض.keras"], compile=False)

    return m1, m2, m3

# تشغيل دالة التحميل وتجهيز المتغيرات (مهم جداً مكانها هنا!)
try:
    m_plant, m_health, m_disease = load_models()
    models_loaded = True
except Exception as e:
    st.error(f"❌ فشل في تحميل الموديلات من السيرفر: {e}")
    models_loaded = False

# --- 3. دالة تجهيز ومعالجة الصور ---
def preprocess_image(uploaded_file, target_size):
    img = Image.open(uploaded_file).convert('RGB')
    img = img.resize(target_size)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0
    return img_array

# --- 4. واجهة المستخدم (Streamlit UI) ---
st.title("🌱 P.L.A.N.T. M.E.D. AI")
st.subheader("نظام التشخيص البيئي والذكاء الاصطناعي المتكامل للأوراق")

tab1, tab2 = st.tabs(["📁 رفع صورة من الجهاز", "📷 (Scan) تصوير مباشر"])
final_image = None

with tab1:
    st.markdown("### 📥 ارفع صورة موجودة مسبقاً على جهازك")
    uploaded_file = st.file_uploader("اختر صورة (JPG, PNG)...", type=["jpg", "jpeg", "png"], key="uploader")
    if uploaded_file is not None:
        final_image = uploaded_file

with tab2:
    st.markdown("### 📸 التقط صورة حية لورقة النبات")
    camera_file = st.camera_input("وجه الكاميرا نحو الورقة واضغط التقاط")
    if camera_file is not None:
        final_image = camera_file

# --- 5. تشغيل الموديلات والتوقعات عند توفر الصورة ---
if final_image is not None:
    st.image(final_image, caption='الصورة التي يتم تحليلها حالياً', use_column_width=True)
    
    if not models_loaded:
        st.error("لا يمكن فحص الصورة لأن الموديلات لم يتم تعريفها بشكل صحيح فوق.")
    else:
        st.info("جاري تحليل الصورة... 🔄")
        try:
            # تجهيز الأبعاد المظبوطة
            img_224 = preprocess_image(final_image, (224, 224))
            img_160 = preprocess_image(final_image, (160, 160))

            # تشغيل التوقعات (الآن المتغيرات معرفة ومضمونة)
            pred_plant = m_plant.predict(img_224)
            pred_health = m_health.predict(img_160)
            pred_disease = m_disease.predict(img_224)

            # عرض المخرجات المبدئية للأرقام
            st.subheader("📊 نتائج التحليل المبدئية:")
            st.write("مخرجات موديل النوع:", pred_plant)
            st.write("مخرجات موديل الصحة:", pred_health)
            st.write("مخرجات موديل الأمراض:", pred_disease)

        except Exception as e:
            st.error(f"حدث خطأ أثناء معالجة الصورة: {e}")
