import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from PIL import Image
import os
import zipfile

# 1. إعداد الصفحة بأعلى جودة احترافية تناسب مشروعك الأكاديمي
st.set_page_config(page_title="P.L.A.N.T. M.E.D. AI", page_icon="🌿", layout="centered")

# 2. هندسة التصميم الاحترافي والـ CSS (UI/UX Premium Design) لتبهر الدكاترة
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
    
    .stButton>button { 
        background: linear-gradient(90deg, #2ecc71 0%, #27ae60 100%); 
        color: white; 
        border-radius: 12px; 
        font-weight: bold; 
        font-size: 16px;
        padding: 10px 24px;
        border: none;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(46, 204, 113, 0.4);
    }
    </style>
    
    <div class="welcome-card">
        <div class="welcome-title">🌿 منصة P.L.A.N.T. M.E.D. AI</div>
        <div class="welcome-subtitle">
            مرحباً بكم دكاترة المستقبل في البروتوكول الرقمي المتكامل للتشخيص البيئي المعتمد على عقول الذكاء الاصطناعي التتابعية لفحص محاصيل النعناع والريحان.
        </div>
    </div>
    
    <div class="ai-engine-container">
        <div class="ai-box">
            <h5>الموديل الأول</h5>
            <p>🧬 تصنيف العائلة النباتية</p>
        </div>
        <div class="ai-box">
            <h5>الموديل الثاني</h5>
            <p>🔍 كاشف الخلايا المصابة</p>
        </div>
        <div class="ai-box">
            <h5>الموديل الثالث</h5>
            <p>📊 بروتوكول العلاج والمكافحة</p>
        </div>
    </div>
""", unsafe_allow_html=True)

# --- 1. دالة فك ضغط مستودع الموديلات بذكاء وديناميكية ---
@st.cache_resource
def extract_and_get_model_path(model_filename):
    zip_filename = "احمد حسني.zip"  # تم تعديل اسم ملف الـ zip المرفوع بملفك بوجود المسافة بين الاسمين
    extract_to = "models_data"
    
    # فك الضغط إذا لم يكن مفكوكاً مسبقاً تسريعاً للتشغيل
    if not os.path.exists(extract_to):
        if os.path.exists(zip_filename):
            with zipfile.ZipFile(zip_filename, 'r') as zip_ref:
                zip_ref.extractall(extract_to)
        else:
            raise FileNotFoundError(f"ملف المستودع المضغوط '{zip_filename}' غير موجود في الخادم الحالي.")
            
    # البحث الديناميكي عن الموديل داخل المجلد لضمان الوصول له أياً كان مساره الداخلي
    for root, dirs, files in os.walk(extract_to):
        if model_filename in files:
            return os.path.join(root, model_filename)
            
    raise FileNotFoundError(f"لم يتم العثور على الموديل: '{model_filename}' داخل المجلد المضغوط.")

# --- 2. دالة استدعاء الموديلات الثلاثة وتخزينها كـ Cache ---
@st.cache_resource
def load_all_models():
    # تم تعديل المسميات بالملي لتتطابق مع ملفاتك المرفوعة من الداخل باللغتين والمسافات
    path_plant = extract_and_get_model_path("Mint vs Basil.keras")
    path_disease = extract_and_get_model_path("موديل الامراض.keras")
    path_health = extract_and_get_model_path("موديل السليم.keras")
    
    return tf.keras.models.load_model(path_plant, compile=False), \
           tf.keras.models.load_model(path_disease, compile=False), \
           tf.keras.models.load_model(path_health, compile=False)

# تشغيل عملية استدعاء النماذج
try:
    model_plant, model_disease, model_health = load_all_models()
except Exception as e:
    st.error(f"❌ خطأ تقني في استدعاء عقول المحاكاة: {e}")

# --- 3. دالة بروتوكول العلاج والمكافحة الزراعية المتكاملة ---
def get_treatment_sheet(disease_index):
    sheet = {
        0: {'عربي': 'البياض الزغبي (Downy Mildew)', 'الأسباب': 'رطوبة نسبية مرتفعة، ونقص تدوير الهواء داخل الصوبة الزراعية.', 'العلاج الطبيعي': 'تقليل فترات الري، وتقليم الأوراق القاعدية المصابة لتحسين التهوية الطبية.', 'العلاج الكيميائي': 'الرش الفوري بمبيدات نحاسية جهازية متخصصة تحت إشراف زراعي.'},
        1: {'عربي': 'تبقع الأوراق الفطري (Leaf Spot)', 'الأسباب': 'تراكم رذاذ الماء على السطح العلوي للأوراق، وزيادة الكثافة النباتية في الأحواض.', 'العلاج الطبيعي': 'تعديل نظام الري ليكون جذرياً (بالتنقيط) وتجنب ري الأوراق علوياً، مع زيادة مسافات الشتل.', 'العلاج الكيميائي': 'مركبات فطرية وقائية واسعة المدى تحتوي على مادة الكابتان النشطة.'},
        2: {'عربي': 'البياض الدقيقي (Powdery Mildew)', 'الأسباب': 'مناخ دافئ وجاف نهاراً مع ارتفاع الرطوبة ليلاً، وغياب الإضاءة والشمس المباشرة.', 'العلاج الطبيعي': 'المكافحة الحيوية باستخدام رش محلول بيكربونات البوتاسيوم أو زيت النيم الطبيعي المستخلص.', 'العلاج الكيميائي': 'استخدام مبيدات فطرية علاجية متخصصة تحتوي على مركب الـ Penconazole.'},
        3: {'عربي': 'صدأ الأوراق الفطري (Mint Rust)', 'الأسباب': 'بيئة مشبعة بالرطوبة العالية الجوية، وانتشار جراثيم فطرية يوريدية برتقالية.', 'العلاج الطبيعي': 'حش كامل المجموع الخضري المصاب فوراً والتخلص منه بالهدم أو الحرق بعيداً عن الصوبة لمنع انتشار الأبواغ الخلوية.', 'العلاج الكيميائي': 'مركبات علاجية جهازية قوية تحتوي على مادة الـ Tebuconazole الفعالة.'}
    }
    return sheet.get(disease_index, {'عربي': 'آفة غير مألوفة', 'الأسباب': 'تحليل مخبري إضافي مطلوب لمعرفة السلالة النادرة.', 'العلاج الطبيعي': 'عزل العينات الفردية المريضة فوراً للحد من العدوى التبادلية.', 'العلاج الكيميائي': 'استشارة لجنة الدعم الفني الزراعي المختصة لإعداد تركيبة مخصصة.'})

# --- 4. دالة معالجة الصورة والتنبؤ التتابعي ---
def process_and_predict(image_data):
    image_display = Image.open(image_data)
    st.image(image_display, caption="📸 العينة الطبية المرفوعة للفحص الفوري في الصوبة", use_container_width=True)
    
    with st.spinner("⏳ جاري تحليل العينة وتشغيل عقول المحاكاة الـ 3 بالتناوب..."):
        try:
            # تجهيز مصفوفة الصورة وحجمها لتطابق شروط إدخال الـ Neural Networks
            img = image_display.resize((224, 224))
            x = image.img_to_array(img)
            x = np.expand_dims(x, axis=0) / 255.0
            
            # الخطوة أ: تصنيف نوع النبات (نعناع أم ريحان)
            plant_preds = model_plant.predict(x, verbose=0)
            detected_plant = "ريحان (Basil)" if np.argmax(plant_preds) == 0 else "نعناع (Mint)"
            
            # الخطوة ب: الفحص المبدئي للصحة (سليم أم مصاب)
            health_preds = model_health.predict(x, verbose=0)
            is_healthy = np.argmax(health_preds) == 0
            
            st.markdown(f"### 📊 النتيجة المخبرية المبدئية للتصنيف: **{detected_plant}**")
            
            if is_healthy:
                st.balloons()  # تأثير مبهج للنجاح
                st.markdown(f"""
                <div class="healthy-box">
                    <h3 style="color: #27ae60; margin-top:0;">✅ تقرير الحالة: عينة طبية سليمة (Healthy Plant)</h3>
                    <p style="color: #34495e; font-size: 15px;">أظهرت التحليلات التوافقية للنماذج أن الأنسجة الخلوية لورقة الـ <b>{detected_plant}</b> سليمة تماماً وتخلو من المسببات الفطرية أو البكتيرية الحالية.</p>
                    <p style="color: #16a085; font-weight: bold; margin-bottom: 0;">🌱 توصية الرعاية: استمر على معدلات التسميد والري الحالية مع المراقبة الدورية.</p>
                </div>
                """, unsafe_allow_html=True)
            else:
                # الخطوة ج: استدعاء نموذج المرض في حال كانت العينة مصابة
                disease_preds = model_disease.predict(x, verbose=0)
                disease_idx = np.argmax(disease_preds)
                info = get_treatment_sheet(disease_idx)
                
                st.markdown(f"""
                <div class="report-box">
                    <h3 style="color: #c0392b; margin-top:0;">🔬 التشخيص السريري: مصاب بـ {info['عربي']}</h3>
                    <hr style="border: 0; border-top: 1px solid #f2d7d5; margin: 10px 0;">
                    <p style="color: #2c3e50;"><b>⚠️ المسببات الإيكولوجية البيئية:</b> {info['الأسباب']}</p>
                    <p style="color: #27ae60;"><b>🌱 بروتوكول المكافحة والوقاية الزراعية:</b> {info['العلاج الطبيعي']}</p>
                    <p style="color: #2980b9;"><b>💊 العلاج الكيميائي والمبيدات الموصى بها:</b> {info['العلاج الكيميائي']}</p>
                    <p style="font-size: 11px; color: #95a5a6; margin-top: 15px; text-align: left;">TRI SCIENCE AI - Integrated Multimodal System © 2026</p>
                </div>
                """, unsafe_allow_html=True)
                
        except Exception as e:
            st.error(f"❌ خطأ تقني في قراءة المصفوفات الخلوية: {e}")

# --- 5. واجهة الفرز والتبويبات الفنية للكاميرا والمعرض ---
tab1, tab2 = st.tabs(["📸 مسح ضوئي حي (Scan)", "📁 استيراد عينة من المعرض"])

with tab1:
    st.subheader("تحليل فوري ومباشر عبر كاميرا المستشعر الذكي")
    camera_photo = st.camera_input("ضع ورقة النبات في منتصف الإطار المخصص للفحص الفوري")
    if camera_photo is not None:
        process_and_predict(camera_photo)

with tab2:
    st.subheader("تحليل ملفات الصور المخزنة مسبقاً من الأجهزة")
    uploaded_file = st.file_uploader("قم برفع ملف الصورة بصيغة مدعومة للحقل الزراعي...", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        process_and_predict(uploaded_file)
