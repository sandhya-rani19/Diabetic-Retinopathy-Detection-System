import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Class index to label mapping
class_labels = {
    0: "Mild",
    1: "Moderate",
    2: "No_DR",
    3: "Proliferative_DR",
    4: "Severe"
}

# Load model
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model("c:/Users/jahna/Downloads/retinopathy_model.h5")
    return model

model = load_model()

# Title
st.title("Diabetic Retinopathy Detection")
st.markdown("Upload a retina image and the model will classify it into one of the following categories:")
st.markdown("- **Severe**\n- **Proliferate_DR**\n- **No_DR**\n- **Moderate**\n- **Mild**")

# Image uploader
uploaded_file = st.file_uploader("Upload an eye image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Uploaded Image",use_container_width=True)

    # Preprocess image
    img_resized = img.resize((150, 150))
    img_array = tf.keras.utils.img_to_array(img_resized) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Predict
    predictions = model.predict(img_array)
    predicted_index = np.argmax(predictions[0])
    confidence = np.max(predictions[0])
    predicted_label = class_labels.get(predicted_index, "Unknown")

    st.success(f"**Prediction:** {predicted_label}")