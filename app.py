import streamlit as st
import os
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import pandas as pd
import traceback

st.set_page_config(page_title="Pneumothorax Classifier", layout="centered")
st.title("Pneumothorax Type Classifier")
st.write("Upload a chest X-ray image to classify it as Simple or Tension Pneumothorax.")

# Create upload folder if not exists
UPLOAD_DIR = "uploaded"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# Try to load model
@st.cache_resource
def load_trained_model():
    try:
        model_path = "pneumothorax_classifier.keras" if os.path.exists("pneumothorax_classifier.keras") else "pneumothorax_classifier.h5"
        model = load_model(model_path)
        return model
    except Exception as e:
        st.error("🚨 Failed to load model.")
        st.text(traceback.format_exc())
        return None

model = load_trained_model()

# Binary labels
class_names = ['No Pneumothorax', 'Pneumothorax']

def predictor(img_path):
    try:
        img = load_img(img_path, target_size=(224, 224))
        img = img_to_array(img)
        img = np.expand_dims(img, axis=0) / 255.0
        prob = model.predict(img)[0][0]
        prediction = {
            'class': class_names[int(prob > 0.5)],
            'confidence': float(prob if prob > 0.5 else 1 - prob)
        }
        return pd.DataFrame([prediction])
    except Exception as e:
        st.error("⚠️ Prediction failed.")
        st.text(traceback.format_exc())
        return None

# File uploader
uploaded_file = st.file_uploader("📤 Upload Chest X-ray", type=["jpg", "jpeg", "png"])

if uploaded_file and model:
    file_path = os.path.join(UPLOAD_DIR, uploaded_file.name)
    with open(file_path, 'wb') as f:
        f.write(uploaded_file.getbuffer())

    st.image(Image.open(file_path).convert("RGB").resize((500, 300)), caption='Uploaded Image')

    prediction = predictor(file_path)
    os.remove(file_path)

    if prediction is not None:
        st.markdown("###  Prediction Result")
        fig, ax = plt.subplots()
        sns.barplot(y='class', x='confidence', data=prediction, ax=ax, palette='Blues_d')
        ax.set(xlabel='Confidence', ylabel='Class')
        st.pyplot(fig)
