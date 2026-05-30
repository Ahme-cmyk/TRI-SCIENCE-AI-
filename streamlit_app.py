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

# تشغيل دالة التحميل وتجهيز المتغيرات
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
        st.error("لا يمكن فحص الصورة لأن الموديلات لم يتم تحميلها بشكل صحيح.")
    else:
        st.info("جاري تحليل الصورة والتشخيص الذكي... 🔄")
        try:
            # تجهيز الأبعاد المظبوطة لكل موديل
            img_224 = preprocess_image(final_image, (224, 224))
            img_160 = preprocess_image(final_image, (160, 160))

            # تشغيل التوقعات
            pred_plant = m_plant.predict(img_224)
            pred_health = m_health.predict(img_160)
            pred_disease = m_disease.predict(img_224)

            # 1. تحديد نوع النبات 
            plant_classes = ["ريحان", "نعناع"]
            plant_index = np.argmax(pred_plant)
            detected_plant = plant_classes[plant_index]
            
            st.success(f"📌 **نوع النبات المكتشف:** {detected_plant}")

            # استخراج القيمة الرقمية لموديل الصحة
            health_value = float(pred_health[0][0])

            # 2. الفصل الذكي والمعدل بدقة بناءً على القراءات الأخيرة لحالات المرض
            if detected_plant == "ريحان":
                # للريحان: الرقم الصغير جداً (0.16) مصاب، والكبير سليم
                is_healthy = health_value > 0.5
            else:
                # للنعناع: تم تعديل العتبة إلى 0.3 ليلقط مرض البياض الدقيقي (0.33) ومرض الصدأ (0.68) كمرضى فوراً
                # النعناع يكون سليم فقط إذا كان أقل من 0.3
                is_healthy = health_value < 0.3
            
            if is_healthy:
                st.balloons()
                st.success(f"💖 **حالة النبات:** سليم ومعافى! (قراءة الصحة: {health_value:.4f})")
            else:
                st.error(f"🚨 **حالة النبات:** مصاب بمرض (قراءة الصحة: {health_value:.4f})، جاري فحص الأعراض...")
                
                # 3. تحديد نوع المرض بناءً على الـ argmax المطلق
                disease_index = np.argmax(pred_disease)
                disease_classes = [
                    "البياض الزغبي (Basil Downy Mildew)", 
                    "تبقع الأوراق (Basil Leaf Spot)", 
                    "البياض الدقيقي (Mint Powdery Mildew)", 
                    "صدأ الأوراق (Mint Rust)"
                ]
                detected_disease = disease_classes[disease_index]
                
                st.warning(f"🔍 **التشخيص الدقيق للمرض:** {detected_disease}")
                
                # 4. قاموس العلاج الشامل
                DISEASES_DATABASE = {
                    "البياض الزغبي (Basil Downy Mildew)": {
                        "fast": "تقليل الرطوبة تماماً حول الريحان، والتخلص فوراً من الأوراق المصابة بشدة وحرقها، وتجنب الري العلوي (رش الأوراق).",
                        "chemical": "الرش بمبيد فطري جهازي يحتوي على مادة (ميتالاكسيل) أو (ميفينوكسام) لحماية الأوراق الجديدة."
                    },
                    "تبقع الأوراق (Basil Leaf Spot)": {
                        "fast": "تهوية النباتات وزيادة المسافات بينها لتجفيف الأوراق بسرعة، وإزالة مخلفات النباتات المصابة من التربة.",
                        "chemical": "استخدام مركبات النحاس الوقائية (مثل كبريتات النحاس أو هيدروكسيد النحاس) بانتظام كل 7-10 أيام."
                    },
                    "البياض الدقيقي (Mint Powdery Mildew)": {
                        "fast": "تقليم الأجزاء الكثيفة لزيادة تغلغل الضوء والهواء، وعزل النعناع المصاب عن باقي النباتات السليمة.",
                        "chemical": "الرش بمبيد فطري يحتوي على مادة (بينكونازول) أو استخدام الكبريت الميكروني المخفف بالماء."
                    },
                    "صدأ الأوراق (Mint Rust)": {
                        "fast": "قص النعناع المصاب بالكامل حتى مستوى سطح التربة (لأنه ينمو مجدداً سريعاً) للتخلص من البثور الفطرية تماماً.",
                        "chemical": "الرش بمبيدات فطرية تحتوي على مواد مثل (بروكونازول) أو (تيزول) لضمان القضاء على جراثيم الصدأ."
                    }
                }
                
                # عرض كروت العلاج للمستخدم
                if detected_disease in DISEASES_DATABASE:
                    st.write("---")
                    col1, col2 = st.columns(2)
                    with col1:
                        st.subheader("⏱️ الإجراء السريع والفوري:")
                        st.info(DISEASES_DATABASE[detected_disease]["fast"])
                    with col2:
                        st.subheader("🧪 العلاج الكيميائي والمبيدات:")
                        st.warning(DISEASES_DATABASE[detected_disease]["chemical"])
                else:
                    st.info("ℹ️ لم يتم إدخال تفاصيل علاج هذا المرض في قاعدة البيانات حتى الآن.")

        except Exception as e:
            st.error(f"حدث خطأ أثناء معالجة الصورة أو استخراج التوقعات: {e}")
