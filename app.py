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
st.title("🩻 Pneumothorax Type Classifier")
st.write("Upload a chest X-ray image to classify it as Pneumothorax or Not. If Pneumothorax is detected, you can classify the type (Simple vs Tension).")

UPLOAD_DIR = "uploaded"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# -------- Load models --------
@st.cache_resource
def load_models():
    try:
        base_model_path = "pneumothorax_classifier.keras" if os.path.exists("pneumothorax_classifier.keras") else "pneumothorax_classifier.h5"
        detailed_model_path = "classification.h5"
        base_model = load_model(base_model_path)
        detailed_model = load_model(detailed_model_path) if os.path.exists(detailed_model_path) else None
        return base_model, detailed_model
    except Exception:
        st.error("🚨 Failed to load one or more models.")
        st.text(traceback.format_exc())
        return None, None

base_model, detailed_model = load_models()

# Class names
binary_classes = ['No Pneumothorax', 'Pneumothorax']
detailed_classes = ['Simple Pneumothorax', 'Tension Pneumothorax']

def predict_with_model(model, img_path, classes):
    try:
        img = load_img(img_path, target_size=(224, 224))
        img = img_to_array(img)
        img = np.expand_dims(img, axis=0) / 255.0
        prob = model.predict(img)[0][0]
        prediction = {
            'class': classes[int(prob > 0.5)],
            'confidence': float(prob if prob > 0.5 else 1 - prob)
        }
        return pd.DataFrame([prediction])
    except Exception:
        st.error("⚠️ Prediction failed.")
        st.text(traceback.format_exc())
        return None

# -------- Upload image --------
uploaded_file = st.file_uploader("📤 Upload Chest X-ray", type=["jpg", "jpeg", "png"])

if uploaded_file and base_model:
    file_path = os.path.join(UPLOAD_DIR, uploaded_file.name)
    with open(file_path, 'wb') as f:
        f.write(uploaded_file.getbuffer())

    st.image(Image.open(file_path).convert("RGB").resize((500, 300)), caption='Uploaded Image')

    prediction = predict_with_model(base_model, file_path, binary_classes)
    os.remove(file_path)

    if prediction is not None:
        st.markdown("### 🧠 Primary Classification Result")
        fig, ax = plt.subplots()
        sns.barplot(y='class', x='confidence', data=prediction, ax=ax, palette='Blues_d')
        ax.set(xlabel='Confidence', ylabel='Class')
        st.pyplot(fig)

        # If Pneumothorax → show second model
        if prediction['class'][0] == "Pneumothorax" and detailed_model:
            st.markdown("### 🔍 Further Classification: Type of Pneumothorax")
            if st.button("Classify Pneumothorax Type"):
                second_result = predict_with_model(detailed_model, file_path, detailed_classes)
                if second_result is not None:
                    st.success(f"Predicted: {second_result['class'][0]}")
                    fig2, ax2 = plt.subplots()
                    sns.barplot(y='class', x='confidence', data=second_result, ax=ax2, palette='Reds')
                    ax2.set(xlabel='Confidence', ylabel='Class')
                    st.pyplot(fig2)
        elif prediction['class'][0] == "Pneumothorax":
            st.warning("🔧 Detailed classification model (classification.h5) not found.")
