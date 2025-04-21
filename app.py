import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np

# Set page config
st.set_page_config(page_title="Monkeypox Skin Lesion Detector", layout="centered")

# Title
st.title("🧪 Monkeypox Skin Lesion Detection")
st.markdown("Upload a skin image to check if it shows signs of **Monkeypox**.")

# Load model
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("monkeypox_model.h5")

model = load_model()

# Function to preprocess image
def preprocess_image(image):
    image = image.resize((256, 256))
    image_array = np.array(image) / 255.0
    if image_array.shape[-1] == 4:  # Convert RGBA to RGB
        image_array = image_array[..., :3]
    return np.expand_dims(image_array, axis=0)

# Upload Image
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

# Predict button
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)
    st.write("Analyzing...")
    
    input_data = preprocess_image(image)
    prediction = model.predict(input_data)[0][0]

    st.subheader("🧬 Prediction Result:")
    if prediction > 0.5:
        st.error(f"🔴 Monkeypox Detected! (Confidence: {prediction:.2f})")
    else:
        st.success(f"🟢 No Monkeypox Detected. (Confidence: {1 - prediction:.2f})")

# Footer
st.markdown("---")
st.markdown("Made with ❤️ using Streamlit and TensorFlow")