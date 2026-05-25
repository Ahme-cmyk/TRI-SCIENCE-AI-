import streamlit as st
import numpy as np

# --- حركة برمجية استراتيجية لعمل Patch لمكتبة Keras قبل استدعاء TensorFlow ---
import sys
import keras

original_dense_init = keras.layers.Dense.__init__
def patched_dense_init(self, *args, **kwargs):
    kwargs.pop('quantization_config', None)
    original_dense_init(self, *args, **kwargs)
keras.layers.Dense.__init__ = patched_dense_init

import tensorflow as tf
from tensorflow.keras.preprocessing import image
from PIL import Image
import os
import zipfile
import urllib.request

# 1. إعداد الصفحة بأعلى جودة احترافية
st.set_page_config(page_title="P.L.A.N.T. M.E.D. AI", page_icon="🌿", layout="centered")

# 2. هندسة التصميم الاحترافي والـ CSS
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Cairo:wght=400;600;800&display=swap');
    
    html, body, [class*="css"] { 
        font-family: 'Cairo', sans-serif; 
        text-align: right; 
        direction: rtl;
    }
    
    .welcome-card {
        background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
        color: white;
        padding: 30px;
        border-radius: 20px;
        box-shadow: 0 10px 20px rgba(0,0,0,0.1);
        text-align: center;
        margin-bottom: 25px;
        border-bottom: 5px solid #2ecc71;
    }
    .welcome-title { font-size: 32px; font-weight: 800; margin-bottom: 10px; }
    .welcome-subtitle { font-size: 16px; opacity: 0.9; line-height: 1.6; }
    
    .ai-engine-container {
        display: flex;
        justify-content: space-around;
        gap: 10px;
        margin-bottom: 25px;
    }
    .ai-box {
        background-color: #ffffff;
        border: 1px solid #e0e0e0;
        border-radius: 12px;
        padding: 15px;
        text-align: center;
        flex: 1;
        box-shadow: 0 4px 6px rgba(0,0,0,0.02);
    }
    .ai-box h5 { color: #2c3e50; margin: 0 0 5px 0; font-weight: 700; }
    .ai-box p { color: #2ecc71; margin: 0; font-size: 13px; font-weight: 600; }
    
    .report-box { 
        background-color: #fdfefe; 
        padding: 25px; 
        border-radius: 18px; 
        border-right: 10px solid #e74c3c; 
        box-shadow: 0 8px 16px rgba(0,0,0,0.05);
        margin-top: 20px; 
    }
    .healthy-box { 
        background-color: #f4fbf7; 
        padding: 25px; 
        border-radius: 18px; 
        border-right: 10px solid #2ecc71; 
        box-shadow: 0 8px 16px rgba(0,0,0,0.05);
        margin-top: 20px; 
    }
    </style>
    
    <div class="welcome-card">
        <div class="welcome-title">🌿 منصة P.L.A.N.T. M.E.D. AI</div>
        <div class="welcome-subtitle">
            مرحباً بكم دكاترة المستقبل في البروتوكول الرقمي المتكامل للتشخيص البيئي المعتمد على عقول الذكاء الاصطناعي التتابعية لفحص محاصيل النعناع والريحان.
        </div>
    </div>
""", unsafe_allow_html=True)

# --- 1. دالة تحميل وفك ضغط ملف الموديلات ---
@st.cache_resource
def download_and_extract_models(model_filename):
    zip_filename = "ahmed_hosny_models.zip"
    extract_to = "models_data"
    hf_download_url = "https://huggingface.co/ahmedhosny2052005/TRI-SCIENCE-AI/resolve/main/%D8%A7%D8%AD%D9%85%D8%AF%20%D8%AD%D8%B3%D9%86%D9%8A.zip"
    
    if not os.path.exists(zip_filename) and not os.path.exists(extract_to):
        with st.spinner('📥 جاري جلب عقول المحاكاة من الخادم السحابي...'):
            urllib.request.urlretrieve(hf_download_url, zip_filename)
            
    if not os.path.exists(extract_to):
        with st.spinner('🔓 جاري فك وحقن الموديلات...'):
            with zipfile.ZipFile(zip_filename, 'r') as zip_ref:
                zip_ref.extractall(extract_to)
                
    for root, dirs, files in os.walk(extract_to):
        if model_filename in files:
            return os.path.join(root, model_filename)
    raise FileNotFoundError(f"لم يتم العثور على الموديل: '{model_filename}'")

# --- 2. استدعاء الموديلات ---
@st.cache_resource
def load_all_models():
    path_plant = download_and_extract_models("Mint vs Basil.keras")
    path_disease = download_and_extract_models("موديل الامراض.keras")
    path_health = download_and_extract_models("موديل السليم.keras")
    
    return tf.keras.models.load_model(path_plant, compile=False), \
           tf.keras.models.load_model(path_disease, compile=False), \
           tf.keras.models.load_model(path_health, compile=False)

try:
    model_plant, model_disease, model_health = load_all_models()
    st.success("تم تفعيل عقول المحاكاة الـ 3 بنجاح استراتيجي! 🎉")
except Exception as e:
    st.error(f"❌ خطأ تقني في استدعاء عقول المحاكاة: {e}")

# --- 3. دالة جدول العلاج ---
def get_treatment_sheet(disease_index):
    sheet = {
        0: {'عربي': 'البياض الزغبي (Downy Mildew)', 'الأسباب': 'رطوبة نسبية مرتفعة، ونقص تدوير الهواء داخل الصوبة الزراعية.', 'العلاج الطبيعي': 'تقليل فترات الري، وتقليم الأوراق القاعدية المصابة لتحسين التهوية الطبية.', 'العلاج الكيميائي': 'الرش الفوري بمبيدات نحاسية جهازية متخصصة.'},
        1: {'عربي': 'تبقع الأوراق الفطري (Leaf Spot)', 'الأسباب': 'تراكم رذاذ الماء على السطح العلوي للأوراق، وزيادة الكثافة النباتية.', 'العلاج الطبيعي': 'تعديل نظام الري ليكون بالتنقيط وتجنب ري الأوراق علوياً.', 'العلاج الكيميائي': 'مركبات فطرية وقائية واسعة المدى تحتوي على مادة الكابتان.'},
        2: {'عربي': 'البياض الدقيقي (Powdery Mildew)', 'الأسباب': 'مناخ دافئ وجاف نهاراً مع ارتفاع الرطوبة ليلاً، وغياب الشمس المباشرة.', 'العلاج الطبيعي': 'المكافحة الحيوية باستخدام رش محلول بيكربونات البوتاسيوم أو زيت النيم.', 'العلاج الكيميائي': 'استخدام مبيدات فطرية علاجية متخصصة تحتوي على مركب الـ Penconazole.'},
        3: {'عربي': 'صدأ الأوراق الفطري (Mint Rust)', 'الأسباب': 'بيئة مشبعة بالرطوبة العالية الجوية، وانتشار جراثيم فطرية.', 'العلاج الطبيعي': 'حش كامل المجموع الخضري المصاب فوراً والتخلص منه بالحرق بعيداً عن الصوبة.', 'العلاج الكيميائي': 'مركبات علاجية جهازية قوية تحتوي على مادة الـ Tebuconazole.'}
    }
    return sheet.get(disease_index, {'عربي': 'آفة غير مألوفة', 'الأسباب': 'تحليل مخبري إضافي مطلوب لمعرفة السلالة.', 'العلاج الطبيعي': 'عزل العينات الفردية المريضة فوراً.', 'العلاج الكيميائي': 'استشارة لجنة الدعم الفني الزراعي المختصة.'})

# --- 4. دالة معالجة الصورة والتنبؤ الدقيق المعاير ---
def process_and_predict(image_data):
    image_display = Image.open(image_data)
    st.image(image_display, caption="📸 العينة المرفوعة للفحص الفوري", use_container_width=True)
    
    with st.spinner("⏳ جاري تفكيك المصفوفات وتحليل العينة بدقة قصوى..."):
        try:
            # تجهيز الصور والمصفوفات
            img_224 = image_display.resize((224, 224))
            x_224 = image.img_to_array(img_224)
            x_224 = np.expand_dims(x_224, axis=0) / 255.0
            
            img_160 = image_display.resize((160, 160))
            x_160 = image.img_to_array(img_160)
            x_160 = np.expand_dims(x_160, axis=0) / 255.0
            
            # --- الفحص الأول: نوع النبات ---
            plant_preds = model_plant.predict(x_224, verbose=0)
            # تطبيق Softmax للتأكد من دقة النسب المئوية
            plant_scores = tf.nn.softmax(plant_preds[0]).numpy()
            detected_plant = "ريحان (Basil)" if np.argmax(plant_scores) == 0 else "نعناع (Mint)"
            plant_conf = np.max(plant_scores) * 100
            
            # --- الفحص الثاني: السليم والمصاب (معايرة ذكية لوقف التخريف) ---
            health_preds = model_health.predict(x_160, verbose=0)
            health_scores = tf.nn.softmax(health_preds[0]).numpy()
            
            # هنا بنجبر الكود يقرأ النسبة الفريش للموديل
            is_healthy = np.argmax(health_scores) == 0
            health_conf = np.max(health_scores) * 100
            
            st.markdown(f"### 📊 تصنيف العائلة النباتية: **{detected_plant}** (درجة التأكيد: {plant_conf:.1f}%)")
            
            # طباعة كواليس الفحص في لوحة تحكم جانبية للمراقبة الأكاديمية
            with st.sidebar:
                st.write("📊 **كواليس عقول المحاكاة (المصفوفات الخام):**")
                st.write(f"نسبة موديل النبات: {plant_scores}")
                st.write(f"نسبة موديل الصحة (0=سليم، 1=مصاب): {health_scores}")

            # لو الموديل مطلع سليم بس النسبة ضعيفة أو الورقة شكلها مصاب، أو لو الموديل نفسه قال مصاب:
            # هنا الكود بيتحقق تماماً من أوزان طبقة التنبؤ
            if is_healthy and health_scores[0] > 0.55:
                st.balloons()
                st.markdown(f"""
                <div class="healthy-box">
                    <h3 style="color: #27ae60; margin-top:0;">✅ تقرير الحالة: عينة طبية سليمة ({health_conf:.1f}%)</h3>
                    <p style="color: #34495e; font-size: 15px;">أظهرت التحليلات التوافقية للنماذج أن الأنسجة الخلوية لورقة الـ <b>{detected_plant}</b> سليمة تماماً وتخلو من المسببات الحالية.</p>
                </div>
                """, unsafe_allow_html=True)
            else:
                # لو الموديل الثاني مطلع سليم بالخطأ، الموديل الثالث (بتاع الأمراض) هو اللي هيحسم الأمر ويكتشف العيب!
                disease_preds = model_disease.predict(x_160, verbose=0)
                disease_scores = tf.nn.softmax(disease_preds[0]).numpy()
                disease_idx = np.argmax(disease_scores)
                disease_conf = np.max(disease_scores) * 100
                
                info = get_treatment_sheet(disease_idx)
                
                st.markdown(f"""
                <div class="report-box">
                    <h3 style="color: #c0392b; margin-top:0;">🔬 التشخيص السريري: مصاب بـ {info['عربي']} ({disease_conf:.1f}%)</h3>
                    <hr style="border: 0; border-top: 1px solid #f2d7d5; margin: 10px 0;">
                    <p style="color: #2c3e50;"><b>⚠️ المسببات الإيكولوجية البيئية:</b> {info['الأسباب']}</p>
                    <p style="color: #27ae60;"><b>🌱 بروتوكول المكافحة والوقاية الزراعية:</b> {info['العلاج الطبيعي']}</p>
                    <p style="color: #2980b9;"><b>💊 العلاج الكيميائي والمبيدات الموصى بها:</b> {info['العلاج الكيميائي']}</p>
                    <p style="font-size: 11px; color: #95a5a6; margin-top: 15px; text-align: left;">TRI SCIENCE AI - Integrated Multimodal System © 2026</p>
                </div>
                """, unsafe_allow_html=True)
                
        except Exception as e:
            st.error(f"❌ خطأ تقني في قراءة المصفوفات الخلوية: {e}")

# --- 5. واجهة الكاميرا والمعرض ---
tab1, tab2 = st.tabs(["📸 مسح ضوئي حي (Scan)", "📁 استيراد عينة من المعرض"])

with tab1:
    st.subheader("تحليل فوري عبر كاميرا المستشعر الذكي")
    camera_photo = st.camera_input("ضع ورقة النبات في منتصف الإطار")
    if camera_photo is not None:
        process_and_predict(camera_photo)

with tab2:
    st.subheader("تحليل ملفات الصور المخزنة")
    uploaded_file = st.file_uploader("قم برفع ملف الصورة...", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        process_and_predict(uploaded_file)
