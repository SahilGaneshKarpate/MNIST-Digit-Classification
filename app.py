import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# ===============================
# Load Trained Model
# ===============================
model = tf.keras.models.load_model("model.h5")

# ===============================
# Streamlit UI
# ===============================
st.set_page_config(
    page_title="MNIST Digit Classification",
    page_icon="🔢",
    layout="centered"
)

st.title("🔢 MNIST Digit Classification")
st.write("Upload a handwritten digit image (28x28 or any image).")

uploaded_file = st.file_uploader(
    "Choose an Image",
    type=["png", "jpg", "jpeg"]
)

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("L")
    st.image(image, caption="Uploaded Image", width=200)

    image = image.resize((28, 28))
    image = np.array(image)

    image = image / 255.0

    image = image.reshape(1, 28, 28)

    prediction = model.predict(image)

    digit = np.argmax(prediction)

    st.success(f"Predicted Digit : {digit}")