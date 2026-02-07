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
    layout="centered",
    initial_sidebar_state="collapsed"
)

# ==================================================
# GLOBAL STYLES (ENTERPRISE UI)
# ==================================================
def inject_css(image_path):
    with open(image_path, "rb") as f:
        encoded = base64.b64encode(f.read()).decode()

    st.markdown(
        f"""
        <style>
        html, body {{
            font-family: 'Inter', sans-serif;
        }}

        .stApp {{
            background:
              linear-gradient(rgba(0,0,0,0.70), rgba(0,0,0,0.70)),
              url("data:image/png;base64,{encoded}");
            background-size: cover;
            background-position: center;
            background-attachment: fixed;
        }}

        .block-container {{
            max-width: 780px;
            background: rgba(255,255,255,0.13);
            backdrop-filter: blur(22px);
            -webkit-backdrop-filter: blur(22px);
            padding: 3.5rem;
            border-radius: 26px;
            border: 1px solid rgba(255,255,255,0.25);
            box-shadow: 0 30px 90px rgba(0,0,0,0.60);
        }}

        /* Buttons */
        button {{
            width: 100%;
            height: 3.4em;
            border-radius: 18px !important;
            font-size: 18px !important;
            font-weight: 600;
            background: linear-gradient(135deg,#00c6ff,#0072ff);
            color: white !important;
            border: none;
        }}

        button:hover {{
            transform: scale(1.03);
            transition: 0.25s ease;
        }}

        /* File uploader */
        section[data-testid="stFileUploader"] {{
            background: rgba(0,0,0,0.45);
            border-radius: 18px;
            padding: 22px;
            border: 1px dashed rgba(255,255,255,0.45);
        }}

        /* Progress bar */
        .stProgress > div > div {{
            background-image: linear-gradient(90deg,#00c6ff,#0072ff);
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

inject_css("assets/watermark.png")

# ==================================================
# SIDEBAR (PRODUCT / ENTERPRISE INFO)
# ==================================================
with st.sidebar:
    st.markdown("## 🐟 Fish AI Platform")
    st.markdown("""
    **Core Technology**
    - SimCLR (Self-Supervised Learning)
    - ResNet50 Deep Encoder
    - Linear Evaluation Protocol

    **Capabilities**
    - Multi-class fish identification
    - High-level feature extraction
    - Robust to lighting & background

    **Use Cases**
    - Fisheries research
    - Education & labs
    - AI demonstrations

    ---
    **Developer**
    **Riad**
    """)

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
# LOAD MODELS (WITH WARM-UP)
# ==================================================
@st.cache_resource(show_spinner=False)
def load_models():
    encoder_path = hf_hub_download(
        repo_id="riad300/fish-simclr-encoder",
        filename="encoder_simclr.pt"
    )

    encoder_state = torch.load(encoder_path, map_location=DEVICE)

    base = models.resnet50(weights=None)
    encoder = nn.Sequential(*list(base.children())[:-1]).to(DEVICE)

    clean_state = {}
    for k, v in encoder_state.items():
        k = k.replace("encoder.", "").replace("backbone.", "").replace("module.", "")
        clean_state[k] = v

    encoder.load_state_dict(clean_state, strict=False)
    encoder.eval()

    classifier = nn.Linear(FEATURE_DIM, NUM_CLASSES)
    classifier.load_state_dict(
        torch.load("models/classifier_final.pt", map_location=DEVICE)
    )
    classifier.to(DEVICE)
    classifier.eval()

    # warm-up
    dummy = torch.randn(1,3,224,224).to(DEVICE)
    with torch.no_grad():
        _ = classifier(encoder(dummy).view(1,-1))

    return encoder, classifier

encoder, classifier = load_models()

# ==================================================
# TRANSFORM
# ==================================================
transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.485,0.456,0.406],
        [0.229,0.224,0.225]
    )
])

# ==================================================
# PREDICTION
# ==================================================
def predict_topk(img, k=3):
    img = transform(img).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        feat = encoder(img).view(1,-1)
        probs = torch.softmax(classifier(feat), dim=1)[0]

    topk = torch.topk(probs, k)
    return [(CLASS_NAMES[i], float(topk.values[idx]*100))
            for idx, i in enumerate(topk.indices)]

# ==================================================
# HEADER
# ==================================================
st.markdown("""
<div style="text-align:center;">
    <h1 style="font-size:46px;">🐟 Fish Species Detection</h1>
    <p style="font-size:18px; color:#dddddd;">
        Industry-Grade AI Fish Classification Platform
    </p>
</div>
<hr style="margin:30px 0;">
""", unsafe_allow_html=True)

# ==================================================
# MAIN APP
# ==================================================
file = st.file_uploader("📤 Upload a fish image", type=["jpg","jpeg","png"])

if file:
    try:
        image = Image.open(file).convert("RGB")
        st.image(image, caption="Uploaded Image", use_column_width=True)

        if st.button("🔍 Analyze Image"):
            with st.spinner("Extracting deep visual features..."):
                results = predict_topk(image)

            st.markdown("## 🧠 Prediction Results")

            for label, conf in results:
                st.markdown(f"**{label}**")
                st.progress(int(conf))
                st.caption(f"Confidence: {conf:.2f}%")

    except Exception:
        st.error("❌ Invalid image. Please upload a valid fish image.")

# ==================================================
# FOOTER
# ==================================================
st.markdown("""
<hr style="margin-top:45px;">
<p style="text-align:center; color:#cfcfcf; font-size:14px;">
© 2026 · Fish AI Classification Platform<br>
Built with PyTorch · SimCLR · Streamlit<br>
Developed by <b>Riad</b>
</p>
""", unsafe_allow_html=True)
