import streamlit as st
import torch
from torchvision import transforms
from PIL import Image
from huggingface_hub import hf_hub_download

# -------------------------
# CONFIG
# -------------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

CLASS_NAMES = [
    "Baim","Bata","Batasio(tenra)","Chitul","Croaker(Poya)",
    "Hilsha","Kajoli","Meni","Pabda","Poli","Puti",
    "Rita","Rui","Rupchada","Silver Carp","Telapiya",
    "carp","k","kaikka","koral","shrimp"
]

ENCODER_URL = "https://huggingface.co/riad300/fish-simclr-encoder/resolve/main/encoder_simclr.pt"
CLASSIFIER_PATH = "models/classifier_final.pt"

# -------------------------
# LOAD MODELS
# -------------------------
@st.cache_resource
def load_models():

    # ─── ENCODER FROM HF URL ─────────────────────────────────────────────
    # download the file once and cache it
    encoder_file = hf_hub_download(
        repo_id="riad300/fish-simclr-encoder",
        filename="encoder_simclr.pt"
    )

    encoder = torch.load(encoder_file, map_location=DEVICE)
    encoder.eval()

    # ─── LOCAL CLASSIFIER ───────────────────────────────────────────────
    classifier = torch.load(CLASSIFIER_PATH, map_location=DEVICE)
    classifier.eval()

    return encoder, classifier


encoder, classifier = load_models()

# -------------------------
# IMAGE TRANSFORMS
# -------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406], [0.229,0.224,0.225])
])

# -------------------------
# PREDICTION
# -------------------------
def predict(img):
    img_t = transform(img).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        feat = encoder(img_t)
        out = classifier(feat)
        probs = torch.softmax(out, dim=1)

    conf, idx = torch.max(probs, dim=1)
    return CLASS_NAMES[idx.item()], conf.item() * 100

# -------------------------
# STREAMLIT UI
# -------------------------
st.set_page_config(page_title="Fish Species Detection", layout="centered")

st.title("🐟 Fish Species Detection & Classification")
st.write("Upload an image and AI will predict the fish species.")

uploaded = st.file_uploader("Choose a fish image", type=["jpg", "jpeg", "png"])

if uploaded:
    image = Image.open(uploaded).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    if st.button("Predict"):
        with st.spinner("Predicting..."):
            label, confidence = predict(image)

        st.success(f"**Prediction:** {label}")
        st.info(f"**Confidence:** {confidence:.2f}%")
