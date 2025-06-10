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



st.title("Pneumothorax Type Classifier")
st.write("Upload a chest X-ray image to classify it as Simple or Tension Pneumothorax.")

# Create upload folder if not exists
if not os.path.exists("uploaded"):
    os.mkdir("uploaded")
# Load the model
model = load_model('pneumothorax_classifier.h5')  # Binary classification model

# Binary labels
class_names = ['No Pneumothorax', 'Pneumothorax']

def predictor(img_path):
    img = load_img(img_path, target_size=(224, 224))  # Match model input
    img = img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = img / 255.0

    prob = model.predict(img)[0][0]
    prediction = {
        'class': class_names[int(prob > 0.5)],
        'confidence': float(prob if prob > 0.5 else 1 - prob)
    }

    df = pd.DataFrame([prediction])
    return df
# Save the uploaded image
def save_uploaded_file(uploaded_file):
    try:
        file_path = os.path.join("uploaded", uploaded_file.name)
        with open(file_path, 'wb') as f:
            f.write(uploaded_file.getbuffer())
        return file_path
    except:
        return None

uploaded_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    file_path = save_uploaded_file(uploaded_file)
    if file_path:
        display_image = Image.open(file_path).convert("RGB")
        st.image(display_image.resize((500, 300)), caption='Uploaded Image', use_column_width=False)

        st.text("Prediction:")
        prediction = predictor(file_path)
        os.remove(file_path)

        fig, ax = plt.subplots()
        sns.barplot(y='class', x='confidence', data=prediction, ax=ax, palette='Blues_d')
        ax.set(xlabel='Confidence %', ylabel='Class')
        st.pyplot(fig)
