import streamlit as st
import torch
import torchvision.transforms as T
from PIL import Image
import json

# -----------------------
# Page Config
# -----------------------
st.set_page_config(
    page_title="🐟 Fish Species Detection",
    page_icon="🐠",
    layout="centered"
)

st.title("🐟 Fish Species Detection System")
st.markdown("**Self-Supervised Learning (SimCLR) based Fish Classification**")

# -----------------------
# Load class names
# -----------------------
with open("models/class_names.json") as f:
    class_names = json.load(f)

# -----------------------
# Load models
# -----------------------
@st.cache_resource
def load_models():
    encoder = torch.load("models/encoder_simclr.pt", map_location="cpu")
    classifier = torch.load("models/classifier.pt", map_location="cpu")
    encoder.eval()
    classifier.eval()
    return encoder, classifier

encoder, classifier = load_models()

# -----------------------
# Image Transform
# -----------------------
transform = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225])
])

# -----------------------
# Upload Image
# -----------------------
uploaded_file = st.file_uploader(
    "📤 Upload a fish image",
    type=["jpg", "png", "jpeg"]
)

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    img_tensor = transform(image).unsqueeze(0)

    with torch.no_grad():
        features = encoder(img_tensor)
        outputs = classifier(features)
        pred = torch.argmax(outputs, dim=1).item()

    st.success(f"🐠 **Predicted Species:** {class_names[str(pred)]}")
