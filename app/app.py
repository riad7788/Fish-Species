import streamlit as st
import os
from PIL import Image
from backend.model_loader import load_models
from backend.predict import predict

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
st.set_page_config(page_title="🐟 Fish Species Classifier", layout="wide")

st.title("Fish Species Classification App")
uploaded_file = st.file_uploader("Upload Fish Image", type=["jpg","png","jpeg"])

# Load models
encoder = torch.load(os.path.join(BASE_DIR, "models", "encoder.pt"), map_location="cpu")
encoder.eval()
classifier, class_names = load_models(BASE_DIR)

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    if st.button("Classify"):
        predicted_class, confidence = predict(image, encoder, classifier, class_names)
        st.success(f"Predicted: {predicted_class} (Confidence: {confidence:.2f})")
