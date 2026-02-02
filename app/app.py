import streamlit as st
from PIL import Image
from utils import send_image

st.set_page_config(page_title="Fish Species Detection", layout="centered")

st.title("🐟 Fish Species Detection using SSL")
st.write("Upload a fish image to identify the species")

uploaded = st.file_uploader("Upload Image", type=["jpg","png","jpeg"])

if uploaded:
    image = Image.open(uploaded).convert("RGB")
    st.image(image, caption="Uploaded Image", width=300)

    if st.button("Predict"):
        with st.spinner("Predicting..."):
            result = send_image(uploaded)

        st.success(f"Prediction: **{result['predicted_species']}**")
        st.write(f"Confidence: {result['confidence']:.2f}")

        st.subheader("Top-3 Predictions")
        for cls, prob in result["top3"]:
            st.write(f"- {cls}: {prob:.2f}")

