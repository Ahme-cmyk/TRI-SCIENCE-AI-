import streamlit as st
import numpy as np
import sys
import keras
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from PIL import Image
import os
import zipfile
import urllib.request

# --- 1. Patching Keras لتجنب أخطاء Serialization ---
original_dense_init = keras.layers.Dense.__init__
def patched_dense_init(self, *args, **kwargs):
    kwargs.pop('quantization_config', None)
    original_dense_init(self, *args, **kwargs)
keras.layers.Dense.__init__ = patched_dense_init

# 2. إعداد واجهة المستخدم
st.set_page_config(page_title="P.L.A.N.T. M.E.D. AI", page_icon="🌿", layout="centered")

st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Cairo:wght@400;600;800&display=swap');
    html, body, [class*="css"] { font-family: 'Cairo', sans-serif; text-align: right; direction: rtl; }
    .welcome-card { background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%); color: white; padding: 30px; border-radius: 20px; box-shadow: 0 10px 20px rgba(0,0,0,0.1); text-align: center; margin-bottom: 25px; border-bottom: 5px solid #2ecc71; }
    .welcome-title { font-size: 32px; font-weight: 800; margin-bottom: 10px; }
    .welcome-subtitle { font-size: 16px; opacity: 0.9; line-height: 1.6; }
    .ai-engine-container { display: flex; justify-content: space-around; gap: 10px; margin-bottom: 25px; }
    .ai-box { background-color: #ffffff; border: 1px solid #e0e0e0; border-radius: 12px; padding: 15px; text-align: center; flex: 1; box-shadow: 0 4px 6px rgba(0,0,0,0.02); }
    .ai-box h5 { color: #2c3e50; margin: 0 0 5px 0; font-weight: 700; }
    .ai-box p { color: #2ecc71; margin: 0; font-size: 13px; font-weight: 600; }
    .report-box { background-color: #fdfefe; padding: 25px; border-radius: 18px; border-right: 10px solid #e74c3c; box-shadow: 0 8px 16px rgba(0,0,0,0.05); margin-top: 20px; }
    .healthy-box { background-color: #f4fbf7; padding: 25px; border-radius: 18px; border-right: 10px solid #2ecc71; box-shadow: 0 8px 16px rgba(0,0,0,0.05); margin-top: 20px; }
    </style>
    <div class="welcome-card">
        <div class="welcome-title">🌿 منصة P.L.A.N.T. M.E.D. AI</div>
        <div class="welcome-subtitle">نظام التشخيص البيئي المعتمد على عقول الذكاء الاصطناعي التتابعية.</div>
    </div>
    <div class="ai-engine-container">
        <div class="ai-box"><h5>الموديل 1</h5><p>🧬 تصنيف النبات</p></div>
        <div class="ai-box"><h5>الموديل 2</h5><p>🔍 كشف الصحة</p></div>
        <div class="ai-box"><h5>الموديل 3</h5><p>📊 العلاج</p></div>
    </div>
""", unsafe_allow_html=True)

# 3. الدوال البرمجية
@st.cache_resource
def download_and_extract_models(model_filename):
    zip_filename = "ahmed_hosny_models.zip"
    extract_to = "models_data"
    hf_download_url = "https://huggingface.co/ahmedhosny2052005/TRI-SCIENCE-AI/resolve/main/%D8%A7%D8%AD%D9%85%D8%AF%20%D8%AD%D8%B3%D9%86%D9%8A.zip"
    if not os.path.exists(extract_to):
        if not os.path.exists(zip_filename): urllib.request.urlretrieve(hf_download_url, zip_filename)
        with zipfile.ZipFile(zip_filename, 'r') as zip_ref: zip_ref.extractall(extract_to)
    for root, dirs, files in os.walk(extract_to):
        if model_filename in files: return os.path.join(root, model_filename)
    raise FileNotFoundError("تعذر تحميل الموديلات")

@st.cache_resource
def load_all_models():
    return tf.keras.models.load_model(download_and_extract_models("Mint vs Basil.keras"), compile=False), \
           tf.keras.models.load_model(download_and_extract_models("موديل الامراض.keras"), compile=False), \
           tf.keras.models.load_model(download_and_extract_models("موديل السليم.keras"), compile=False)

try:
    model_plant, model_disease, model_health = load_all_models()
    st.success("✅ تم تفعيل العقول الثلاثة بنجاح!")
except Exception as e: st.error(f"❌ خطأ تحميل: {e}")

def get_treatment_sheet(idx):
    sheet = {
        0: {'عربي': 'البياض الزغبي', 'الأسباب': 'رطوبة عالية.', 'العلاج الطبيعي': 'تحسين التهوية.', 'العلاج الكيميائي': 'مبيدات نحاسية.'},
        1: {'عربي': 'تبقع الأوراق', 'الأسباب': 'ري علوي.', 'العلاج الطبيعي': 'ري تنقيط.', 'العلاج الكيميائي': 'كابتان.'},
        2: {'عربي': 'البياض الدقيقي', 'الأسباب': 'تذبذب الحرارة.', 'العلاج الطبيعي': 'زيت النيم.', 'العلاج الكيميائي': 'Penconazole.'},
        3: {'عربي': 'صدأ الأوراق', 'الأسباب': 'رطوبة جوية.', 'العلاج الطبيعي': 'حش المصاب.', 'العلاج الكيميائي': 'Tebuconazole.'}
    }
    return sheet.get(idx, {'عربي': 'غير محدد', 'الأسباب': 'فحص مخبري.', 'العلاج الطبيعي': 'عزل العينة.', 'العلاج الكيميائي': 'استشارة فنية.'})

# 4. التنبؤ وتعديل المصفوفات الخام
def process_and_predict(image_data):
    image_display = Image.open(image_data)
    st.image(image_display, caption="📸 العينة", use_container_width=True)
    with st.spinner("🔍 جاري التحليل..."):
        try:
            img_224 = image_display.resize((224, 224))
            x_224 = np.expand_dims(image.img_to_array(img_224), axis=0) / 255.0
            
            img_160 = image_display.resize((160, 160))
            x_160 = np.expand_dims(image.img_to_array(img_160), axis=0) / 255.0
            
            plant_preds = model_plant.predict(x_224, verbose=0)
            plant_scores = tf.nn.softmax(plant_preds[0]).numpy()
            detected_plant = "ريحان (Basil)" if np.argmax(plant_scores) == 0 else "نعناع (Mint)"
            plant_conf = np.max(plant_scores) * 100
            
            # الفحص الثاني (تصحيح Sigmoid)
            health_preds = model_health.predict(x_160, verbose=0)
            raw_health_value = health_preds[0][0]
           if raw_health_value < 0.2: 
    is_healthy = True
    * 100
            else:
                is_healthy = False
                health_conf = raw_health_value * 100
            
            st.markdown(f"### 📊 تصنيف: **{detected_plant}** ({plant_conf:.1f}%)")
            
            with st.sidebar:
                st.write("📊 **كواليس عقول المحاكاة:**")
                st.write(f"الرقم الخام للصحة: {raw_health_value:.4f}")
                st.write(f"القرار: {'سليم' if is_healthy else 'مصاب'}")

            if is_healthy:
                st.balloons()
                st.markdown('<div class="healthy-box"><h3>✅ عينة طبية سليمة</h3></div>', unsafe_allow_html=True)
            else:
                disease_preds = model_disease.predict(x_160, verbose=0)
                disease_idx = np.argmax(tf.nn.softmax(disease_preds[0]))
                info = get_treatment_sheet(disease_idx)
                st.markdown(f"""
                <div class="report-box">
                    <h3>🔬 التشخيص: {info['عربي']}</h3>
                    <p><b>الأسباب:</b> {info['الأسباب']}</p>
                    <p><b>طبيعي:</b> {info['العلاج الطبيعي']}</p>
                    <p><b>كيميائي:</b> {info['العلاج الكيميائي']}</p>
                </div>
                """, unsafe_allow_html=True)
        except Exception as e: st.error(f"⚠️ خطأ: {e}")

# 5. التبويبات
tab1, tab2 = st.tabs(["📸 مسح ضوئي حي", "📁 استيراد من المعرض"])
with tab1:
    photo = st.camera_input("التقط صورة")
    if photo: process_and_predict(photo)
with tab2:
    up = st.file_uploader("ارفع صورة", type=["jpg", "png"])
    if up: process_and_predict(up)
