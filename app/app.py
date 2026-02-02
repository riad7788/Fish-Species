import streamlit as st
import torch
from PIL import Image
import torchvision.transforms as transforms
import numpy as np

# --------------------------------
# Streamlit Page Config
# --------------------------------
st.set_page_config(
    page_title="🐟 Fish Species Classifier",
    page_icon="🐠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --------------------------------
# Sidebar
# --------------------------------
st.sidebar.title("Settings")
st.sidebar.markdown("""
This app classifies fish species using a pretrained SimCLR encoder.
Upload a fish image to see the prediction.
""")

confidence_threshold = st.sidebar.slider("Confidence threshold", 0.0, 1.0, 0.5)

with st.sidebar.expander("About"):
    st.write("""
    - Developed using PyTorch + Streamlit  
    - Features: Upload image, feature extraction, prediction  
    - Model: SimCLR encoder (Fish Species)  
    """)

# --------------------------------
# Main UI Header
# --------------------------------
st.title("🐟 Fish Species Classification App")
st.markdown("Upload an image of a fish and get the predicted species!")

# --------------------------------
# Load Model (cached)
# --------------------------------
@st.cache_resource
def load_model():
    model = torch.load("models/encoder_simclr.pt", map_location="cpu")
    model.eval()
    return model

model = load_model()
st.success("✅ Model loaded successfully!")

# --------------------------------
# Image Upload
# --------------------------------
uploaded_file = st.file_uploader(
    "Upload Fish Image (jpg, png, jpeg)", 
    type=["jpg", "png", "jpeg"],
    accept_multiple_files=False
)

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # --------------------------------
    # Preprocess Image
    # --------------------------------
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])
    img_tensor = transform(image).unsqueeze(0)

    # --------------------------------
    # Columns for Buttons
    # --------------------------------
    col1, col2 = st.columns(2)
    classify_btn = col1.button("🟢 Classify")
    clear_btn = col2.button("🔴 Clear")

    # --------------------------------
    # Prediction
    # --------------------------------
    if classify_btn:
        with torch.no_grad():
            output = model(img_tensor)
        
        # Fake example for demo (replace with actual classifier)
        # output = torch.rand(1,3)  # 3 classes example
        # probs = torch.softmax(output, dim=1).numpy()[0]
        # class_idx = np.argmax(probs)
        # confidence = probs[class_idx]

        st.subheader("Prediction Result")
        st.success("Predicted: Tilapia (Confidence: 0.92)")  # Replace with actual result

    if clear_btn:
        st.experimental_rerun()
