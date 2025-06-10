import streamlit as st
import os
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from helper import predictor

st.title("Pneumothorax Type Classifier")
st.write("Upload a chest X-ray image to classify it as Simple or Tension Pneumothorax.")

# Ensure upload directory exists
if not os.path.exists("uploaded"):
    os.mkdir("uploaded")

# Save uploaded file
def save_uploaded_file(uploaded_file):
    file_path = os.path.join("uploaded", uploaded_file.name)
    with open(file_path, 'wb') as f:
        f.write(uploaded_file.getbuffer())
    return file_path

uploaded_file = st.file_uploader("Upload Chest X-ray", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    file_path = save_uploaded_file(uploaded_file)
    if file_path:
        display_image = Image.open(file_path).convert("RGB")
        st.image(display_image.resize((500, 300)), caption='Uploaded Image', use_container_width=True)

        try:
            prediction = predictor(file_path)
            os.remove(file_path)

            st.markdown("### Prediction Results")
            fig, ax = plt.subplots()
            sns.barplot(y='class', x='confidence', data=prediction, ax=ax, palette='Blues_d')
            ax.set(xlabel='Confidence', ylabel='Class')
            st.pyplot(fig)

        except Exception as e:
            st.error(f"Prediction failed: {str(e)}")
