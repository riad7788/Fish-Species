import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
from huggingface_hub import hf_hub_download
import base64

# ==================================================
# PAGE CONFIG
# ==================================================
st.set_page_config(
    page_title="Fish Species Detection",
    page_icon="🐟",
    layout="centered"
)

# ==================================================
# BACKGROUND + GLASS UI
# ==================================================
def add_bg(image_path):
    with open(image_path, "rb") as f:
        encoded = base64.b64encode(f.read()).decode()

    st.markdown(
        f"""
        <style>
        .stApp {{
            background:
              linear-gradient(rgba(0,0,0,0.55), rgba(0,0,0,0.55)),
              url("data:image/png;base64,{encoded}");
            background-size: cover;
            background-position: center;
            background-attachment: fixed;
        }}

        .block-container {{
            max-width: 720px;
            background: rgba(255, 255, 255, 0.15);
            backdrop-filter: blur(18px);
            -webkit-backdrop-filter: blur(18px);
            padding: 3rem;
            border-radius: 22px;
            border: 1px solid rgba(255,255,255,0.25);
            box-shadow: 0 20px 60px rgba(0,0,0,0.45);
        }}

        /* Buttons */
        button {{
            width: 100%;
            height: 3.2em;
            border-radius: 14px !important;
            font-size: 18px !important;
            font-weight: 600;
            background: linear-gradient(135deg,#00c6ff,#0072ff);
            color: white !important;
            border: none;
        }}

        button:hover {{
            transform: scale(1.02);
            transition: 0.2s ease;
        }}

        /* File uploader */
        section[data-testid="stFileUploader"] {{
            background: rgba(0,0,0,0.35);
            border-radius: 14px;
            padding: 18px;
            border: 1px dashed rgba(255,255,255,0.4);
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

add_bg("assets/watermark.png")

# ==================================================
# CONFIG
# ==================================================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

CLASS_NAMES = [
    "Baim","Bata","Batasio(tenra)","Chitul","Croaker(Poya)",
    "Hilsha","Kajoli","Meni","Pabda","Poli","Puti",
    "Rita","Rui","Rupchada","Silver Carp","Telapiya",
    "carp","k","kaikka","koral","shrimp"
]

NUM_CLASSES = len(CLASS_NAMES)
FEATURE_DIM = 2048

# ==================================================
# LOAD MODELS
# ==================================================
@st.cache_resource
def load_models():

    # -------- Encoder (SimCLR ResNet50) --------
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

    # -------- Classifier --------
    classifier = nn.Linear(FEATURE_DIM, NUM_CLASSES)
    classifier.load_state_dict(
        torch.load("models/classifier_final.pt", map_location=DEVICE)
    )
    classifier.to(DEVICE)
    classifier.eval()

    return encoder, classifier

encoder, classifier = load_models()

# ==================================================
# IMAGE TRANSFORM
# ==================================================
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225]
    )
])

# ==================================================
# PREDICTION
# ==================================================
def predict(img):
    img = transform(img).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        feat = encoder(img)
        feat = feat.view(feat.size(0), -1)
        out = classifier(feat)
        prob = torch.softmax(out, dim=1)

    idx = prob.argmax(1).item()
    return CLASS_NAMES[idx], prob[0][idx].item() * 100

# ==================================================
# UI
# ==================================================
st.markdown("""
<div style="text-align:center;">
    <h1 style="font-size:42px; margin-bottom:5px;">🐟 Fish Species Detection</h1>
    <p style="font-size:18px; color:#e0e0e0;">
        AI-powered Fish Classification using SimCLR & ResNet50
    </p>
</div>
<hr style="margin-top:25px; margin-bottom:30px;">
""", unsafe_allow_html=True)

file = st.file_uploader(
    "📤 Upload a fish image",
    type=["jpg", "jpeg", "png"]
)

if file:
    image = Image.open(file).convert("RGB")
    st.image(image, caption="Uploaded Fish Image", use_column_width=True)

    st.markdown("<br>", unsafe_allow_html=True)

    if st.button("🔍 Predict Species"):
        with st.spinner("Analyzing image..."):
            label, conf = predict(image)

        st.markdown(
            f"""
            <div style="
                margin-top:30px;
                padding:25px;
                border-radius:18px;
                background: linear-gradient(
                    135deg,
                    rgba(0,114,255,0.25),
                    rgba(0,198,255,0.25)
                );
                text-align:center;
                box-shadow: inset 0 0 25px rgba(255,255,255,0.15);
            ">
                <p style="font-size:14px; letter-spacing:1px;">PREDICTED SPECIES</p>
                <h2 style="font-size:34px;">{label}</h2>
                <p style="font-size:18px;">
                    Confidence: <b>{conf:.2f}%</b>
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )

# ==================================================
# FOOTER
# ==================================================
st.markdown("""
<hr style="margin-top:40px;">
<p style="text-align:center; color:#ccc; font-size:14px;">
© 2026 · Fish AI Classification System<br>
Developed by <b>Riad</b>
</p>
""", unsafe_allow_html=True)
