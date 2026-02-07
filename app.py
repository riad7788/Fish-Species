import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
from huggingface_hub import hf_hub_download
import base64

# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------
st.set_page_config(
    page_title="Fish Species Detection",
    page_icon="🐟",
    layout="centered"
)

# --------------------------------------------------
# BACKGROUND + WATERMARK
# --------------------------------------------------
def add_bg(image_path):
    with open(image_path, "rb") as f:
        encoded = base64.b64encode(f.read()).decode()

    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("data:image/png;base64,{encoded}");
            background-size: cover;
            background-position: center;
            background-repeat: no-repeat;
        }}

        .block-container {{
            background-color: rgba(255, 255, 255, 0.88);
            padding: 2.5rem;
            border-radius: 18px;
            box-shadow: 0 10px 40px rgba(0,0,0,0.15);
        }}

        button {{
            border-radius: 12px !important;
            height: 3em;
            font-size: 18px !important;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

add_bg("assets/watermark.png")

# --------------------------------------------------
# CONFIG
# --------------------------------------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

CLASS_NAMES = [
    "Baim","Bata","Batasio(tenra)","Chitul","Croaker(Poya)",
    "Hilsha","Kajoli","Meni","Pabda","Poli","Puti",
    "Rita","Rui","Rupchada","Silver Carp","Telapiya",
    "carp","k","kaikka","koral","shrimp"
]

NUM_CLASSES = len(CLASS_NAMES)
FEATURE_DIM = 2048

# --------------------------------------------------
# LOAD MODELS
# --------------------------------------------------
@st.cache_resource
def load_models():

    # -------- ENCODER --------
    encoder_path = hf_hub_download(
        repo_id="riad300/fish-simclr-encoder",
        filename="encoder_simclr.pt"
    )

    encoder_state = torch.load(encoder_path, map_location=DEVICE)

    base = models.resnet50(weights=None)
    encoder = nn.Sequential(*list(base.children())[:-1])
    encoder.to(DEVICE)

    clean_state = {}
    for k, v in encoder_state.items():
        k = k.replace("encoder.", "").replace("backbone.", "").replace("module.", "")
        clean_state[k] = v

    encoder.load_state_dict(clean_state, strict=False)
    encoder.eval()

    # -------- CLASSIFIER --------
    classifier = nn.Linear(FEATURE_DIM, NUM_CLASSES)
    classifier.load_state_dict(
        torch.load("models/classifier_final.pt", map_location=DEVICE)
    )
    classifier.to(DEVICE)
    classifier.eval()

    return encoder, classifier

encoder, classifier = load_models()

# --------------------------------------------------
# IMAGE TRANSFORM
# --------------------------------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225]
    )
])

# --------------------------------------------------
# PREDICT FUNCTION
# --------------------------------------------------
def predict(img):
    img = transform(img).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        feat = encoder(img)
        feat = feat.view(feat.size(0), -1)
        out = classifier(feat)
        prob = torch.softmax(out, dim=1)

    idx = prob.argmax(1).item()
    return CLASS_NAMES[idx], prob[0][idx].item() * 100

# --------------------------------------------------
# UI
# --------------------------------------------------
st.markdown("""
<h1 style='text-align:center;'>🐟 Fish Species Detection</h1>
<p style='text-align:center; color:gray; font-size:17px;'>
AI-powered Fish Classification using SimCLR + ResNet50
</p>
<hr>
""", unsafe_allow_html=True)

file = st.file_uploader(
    "📤 Upload a fish image",
    type=["jpg", "jpeg", "png"]
)

if file:
    image = Image.open(file).convert("RGB")
    st.image(image, caption="Uploaded Fish Image", use_column_width=True)

    st.markdown("<br>", unsafe_allow_html=True)

    if st.button("🔍 Predict Species", use_container_width=True):
        with st.spinner("Analyzing image..."):
            label, conf = predict(image)

        st.markdown(
            f"""
            <div style="
                background: rgba(0, 136, 204, 0.12);
                padding: 25px;
                border-radius: 15px;
                text-align: center;
                margin-top: 25px;
            ">
                <h3>🐠 Predicted Species</h3>
                <h2 style="color:#007acc;">{label}</h2>
                <p style="font-size:18px;">Confidence: <b>{conf:.2f}%</b></p>
            </div>
            """,
            unsafe_allow_html=True
        )

# --------------------------------------------------
# FOOTER
# --------------------------------------------------
st.markdown("""
<hr>
<p style='text-align:center; color:gray; font-size:14px;'>
© 2026 | Fish AI Classification System <br>
Developed by Riad
</p>
""", unsafe_allow_html=True)
