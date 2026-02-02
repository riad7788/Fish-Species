# app.py - Professional Fish Species UI
import streamlit as st
from PIL import Image

# ----------------------------------------
# Streamlit Page Config
# ----------------------------------------
st.set_page_config(
    page_title="🐟 Fish Species Classifier",
    page_icon="🐠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ----------------------------------------
# Sidebar
# ----------------------------------------
st.sidebar.title("Settings")
st.sidebar.markdown("""
Upload a fish image and see the predicted species.
""")
confidence_threshold = st.sidebar.slider("Confidence threshold", 0.0, 1.0, 0.5)

with st.sidebar.expander("About"):
    st.write("""
    - Developed using PyTorch + Streamlit  
    - Features: Upload image, feature extraction, prediction  
    - Model: SimCLR encoder (Fish Species)  
    """)

# ----------------------------------------
# Main UI
# ----------------------------------------
st.title("🐟 Fish Species Classification App")
st.markdown("Upload an image of a fish and get the predicted species!")

# ----------------------------------------
# Streamlit File Uploader
# ----------------------------------------
uploaded_file = st.file_uploader(
    "Upload Fish Image (jpg, png, jpeg)", 
    type=["jpg", "png", "jpeg"]
)

# ----------------------------------------
# Demo Prediction (No torch required)
# ----------------------------------------
if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Columns for buttons
    col1, col2 = st.columns(2)
    classify_btn = col1.button("🟢 Classify")
    clear_btn = col2.button("🔴 Clear")

    if classify_btn:
        # Fake demo prediction
        st.subheader("Prediction Result")
        st.success("Predicted: Tilapia (Confidence: 0.92)")  # Replace later with actual model

    if clear_btn:
        st.experimental_rerun()
