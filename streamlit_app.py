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

# --- 1. دالة تحميل وفك ضغط ملف الموديلات من الخادم السحابي ---
