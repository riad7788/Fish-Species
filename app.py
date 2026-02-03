import streamlit as st
import torch
import torchvision.transforms as T
from PIL import Image
import json
import os

# ---------------- Page Config ----------------
st.set_page_config(
    page_title="🐟 Fish Species Detection",
    page_icon="🐠",
    layout="centered"
)

st.title("🐟 Fish Species Detection System")
st.markdown("Self-Supervised Learning (**SimCLR**) based Fish Classification")

# ---------------- Sidebar ----------------
st.sidebar.title("📌 Project Info")
st.sidebar.markdown("""
- Course: Capstone  
- Method: SimCLR (SSL)  
- Framework: PyTorch  
- Web App: Streamlit  
- Developer: Riad
""")

# ---------------- Load Class Names ----------------
with open("models/class_names.json") as f:
    class_names = json.load(f)

# ---------------- Load Models from HuggingFace ----------------
@st.cache_resource
def load_models():
    encoder = torch.hub.load_state_dict_from_url(
        "https://huggingface.co/riad300/fish-simclr-encoder/resolve/main/encoder_simclr.pt",
        map_location="cpu"
    )
    encoder.eval()

    classifier = torch.load("models/classifier.pt", map_location="cpu")
    classifier.eval()

    return encoder, classifier

encoder, classifier = load_models()

# ---------------- Image Transform ----------------
transform = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# ---------------- Upload ----------------
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

        pred = torch.argmax(probs, dim=1).item()
        confidence = probs[0][pred].item() * 100

    st.success(f"🐠 **Predicted Species:** {class_names[str(pred)]}")
    st.info(f"🎯 Confidence: {confidence:.2f}%")

st.markdown("---")
st.markdown(
    "<center>© 2026 | Fish Species Detection using SimCLR (SSL)</center>",
    unsafe_allow_html=True
)
