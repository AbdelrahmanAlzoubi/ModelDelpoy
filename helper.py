import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import pandas as pd

# Load your binary classification model
model = load_model('pneumothorax_classifier.h5')

# Class labels: 0 = No Pneumothorax, 1 = Pneumothorax
class_names = ['No Pneumothorax', 'Pneumothorax']

def predictor(img_path):
    img = load_img(img_path, target_size=(224, 224))
    img = img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = img / 255.0

    prob = model.predict(img)[0][0]  # sigmoid → one float value
    predicted_class = int(prob >= 0.5)

    # Confidence for the predicted class
    confidence = prob if predicted_class == 1 else 1 - prob

    preds_df = pd.DataFrame({
        "class": [class_names[predicted_class]],
        "confidence": [confidence]
    })

    return preds_df
