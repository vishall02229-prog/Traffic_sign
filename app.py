import os
import numpy as np
from PIL import Image
import streamlit as st
import tensorflow as tf

st.set_page_config(page_title="Traffic Sign Classifier", page_icon="🚦", layout="centered")

MODEL_PATH = "Traffic.h5"

CLASS_NAMES = [
    "Speed limit (20km/h)", "Speed limit (30km/h)", "Speed limit (50km/h)",
    "Speed limit (60km/h)", "Speed limit (70km/h)", "Speed limit (80km/h)",
    "End of speed limit (80km/h)", "Speed limit (100km/h)", "Speed limit (120km/h)",
    "No passing", "No passing for vehicles over 3.5 tons", "Right-of-way at intersection",
    "Priority road", "Yield", "Stop", "No vehicles", "Vehicles over 3.5 tons prohibited",
    "No entry", "General caution", "Dangerous curve left", "Dangerous curve right",
    "Double curve", "Bumpy road", "Slippery road", "Road narrows on the right",
    "Road work", "Traffic signals", "Pedestrians", "Children crossing",
    "Bicycles crossing", "Beware of ice/snow", "Wild animals crossing",
    "End of all speed and passing limits", "Turn right ahead", "Turn left ahead",
    "Ahead only", "Go straight or right", "Go straight or left", "Keep right",
    "Keep left", "Roundabout mandatory", "End of no passing",
    "End of no passing by vehicles over 3.5 tons"
]

@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"{MODEL_PATH} was not found in the project folder.")
    # compile=False avoids loading the old optimizer state from the original H5 file.
    return tf.keras.models.load_model(MODEL_PATH, compile=False)


def predict_image(image: Image.Image):
    # The original training/prediction pipeline used 30x30 RGB arrays scaled to 0-1.
    image = image.convert("RGB").resize((30, 30))
    array = np.asarray(image, dtype=np.float32) / 255.0
    array = np.expand_dims(array, axis=0)
    probabilities = load_model().predict(array, verbose=0)[0]
    class_id = int(np.argmax(probabilities))
    confidence = float(probabilities[class_id])
    return class_id, CLASS_NAMES[class_id], confidence, probabilities

st.title("🚦 Traffic Sign Classification")
st.write("Upload a traffic-sign image and the trained deep-learning model will predict its class.")

uploaded_file = st.file_uploader("Choose a traffic-sign image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded image", use_container_width=True)

    if st.button("🔍 Predict Traffic Sign", type="primary", use_container_width=True):
        with st.spinner("Analyzing image..."):
            try:
                class_id, label, confidence, probabilities = predict_image(image)
                st.success(f"Prediction: {label}")
                st.metric("Confidence", f"{confidence * 100:.2f}%")
                st.caption(f"Class ID: {class_id}")

                top_indices = np.argsort(probabilities)[-5:][::-1]
                st.subheader("Top 5 predictions")
                for idx in top_indices:
                    st.write(f"**{CLASS_NAMES[int(idx)]}** — {probabilities[int(idx)] * 100:.2f}%")
            except Exception as exc:
                st.error("The model could not process this image.")
                st.exception(exc)

st.divider()
st.caption("Model: Traffic.h5 • Input size: 30×30 • Classes: 43 • Framework: TensorFlow/Keras")
