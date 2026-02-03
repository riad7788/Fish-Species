import streamlit as st
import torch
import torchvision.transforms as T
from PIL import Image
import json
import os
import gdown

# --------------------------------------------------
# Page Config
# --------------------------------------------------
st.set_page_config(
    page_title="🐟 Fish Species Detection",
    page_icon="🐠",
    layout="centered"
)

st.title("🐟 Fish Species Detection System")
st.markdown("Self-Supervised Learning (**SimCLR**) based Fish Classification")

# --------------------------------------------------
# Sidebar
# --------------------------------------------------
st.sidebar.title("📌 Project Info")
st.sidebar.markdown("""
- **Course:** Capstone  
- **Method:** SimCLR (SSL)  
- **Framework:** PyTorch  
- **Web App:** Streamlit  
- **Developer:** Riad
""")

# --------------------------------------------------
# Download models from Google Drive (FIXED)
# --------------------------------------------------
def download_models():
    os.makedirs("models", exist_ok=True)

    encoder_path = "models/encoder_simclr.pt"
    classifier_path = "models/classifier.pt"

    # 🔹 FULL Google Drive link (important)
    encoder_url = (
        "https://drive.google.com/file/d/"
        "1xQYBp_JHVv0MjRpU4WbRphcNHU3NgyR/view"
    )

    if not os.path.exists(encoder_path):
        with st.spinner("⬇️ Downloading encoder model (first run only)..."):
            gdown.download(
                encoder_url,
                encoder_path,
                quiet=False,
                fuzzy=True   # 🔥 THIS IS THE KEY FIX
            )

    if not os.path.exists(classifier_path):
        st.error("❌ classifier.pt not found in models folder!")
        st.stop()

# --------------------------------------------------
# Load class names
# --------------------------------------------------
with open("models/class_names.json", "r") as f:
    class_names = json.load(f)

# --------------------------------------------------
# Load models
# --------------------------------------------------
@st.cache_resource
def load_models():
    encoder = torch.load("models/encoder_simclr.pt", map_location="cpu")
    classifier = torch.load("models/classifier.pt", map_location="cpu")

    encoder.eval()
    classifier.eval()

    return encoder, classifier

# --------------------------------------------------
# Prepare models
# --------------------------------------------------
download_models()
encoder, classifier = load_models()

# --------------------------------------------------
# Image transform
# --------------------------------------------------
transform = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# --------------------------------------------------
# Image upload
# --------------------------------------------------
uploaded_file = st.file_uploader(
    "📤 Upload a fish image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    img_tensor = transform(image).unsqueeze(0)

    with torch.no_grad():
        features = encoder(img_tensor)
        outputs = classifier(features)
        probs = torch.softmax(outputs, dim=1)

        pred_idx = torch.argmax(probs, dim=1).item()
        confidence = probs[0][pred_idx].item() * 100

    st.success(f"🐠 **Predicted Species:** {class_names[str(pred_idx)]}")
    st.info(f"🎯 **Confidence:** {confidence:.2f}%")

# --------------------------------------------------
# Footer
# --------------------------------------------------
st.markdown("---")
st.markdown(
    "<center>© 2026 | Fish Species Detection using SimCLR (SSL)</center>",
    unsafe_allow_html=True
)
