import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from PIL import Image
import os
import gdown
import zipfile
import keras

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

    m1 = tf.keras.models.load_model(
        found_paths["Mint vs Basil.keras"],
        compile=False
    )

    m2 = tf.keras.models.load_model(
        found_paths["موديل السليم.keras"],
        compile=False
    )

    m3 = tf.keras.models.load_model(
        found_paths["موديل الامراض.keras"],
        compile=False
    )

    return m1, m2, m3

m_plant, m_health, m_disease = load_models()

st.write("Plant Shape:", m_plant.input_shape)
st.write("Health Shape:", m_health.input_shape)
st.write("Disease Shape:", m_disease.input_shape)
