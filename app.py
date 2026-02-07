import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms, models
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

# -------------------------
# ENCODER ARCHITECTURE
# -------------------------
class SimCLREncoder(nn.Module):
    def __init__(self):
        super().__init__()
        base = models.resnet50(weights=None)
        self.encoder = nn.Sequential(*list(base.children())[:-1])

    def forward(self, x):
        x = self.encoder(x)
        return x.view(x.size(0), -1)  # [B, 2048]

# -------------------------
# LOAD MODELS
# -------------------------
@st.cache_resource
def load_models():

    # ⬇️ Download encoder weights from HF
    encoder_path = hf_hub_download(
        repo_id="riad300/fish-simclr-encoder",
        filename="encoder_simclr.pt"
    )

    encoder = SimCLREncoder()
    encoder.load_state_dict(
        torch.load(encoder_path, map_location=DEVICE)
    )
    encoder.to(DEVICE)
    encoder.eval()

    # ⬇️ Load classifier (trained on top of encoder)
    classifier = torch.load(
        "models/classifier_final.pt",
        map_location=DEVICE
    )
    classifier.to(DEVICE)
    classifier.eval()

    return encoder, classifier


encoder, classifier = load_models()

# -------------------------
# IMAGE TRANSFORM
# -------------------------
transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.485,0.456,0.406],
        [0.229,0.224,0.225]
    )
])

# -------------------------
# PREDICT
# -------------------------
def predict(img):
    img = transform(img).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        feat = encoder(img)
        out = classifier(feat)
        prob = torch.softmax(out, dim=1)

    idx = prob.argmax(1).item()
    return CLASS_NAMES[idx], prob[0][idx].item() * 100

# -------------------------
# STREAMLIT UI
# -------------------------
st.set_page_config("Fish Species Detection")

st.title("🐟 Fish Species Detection & Classification")

file = st.file_uploader("Upload Fish Image", ["jpg","png","jpeg"])

if file:
    img = Image.open(file).convert("RGB")
    st.image(img, use_column_width=True)

    if st.button("Predict"):
        label, conf = predict(img)
        st.success(f"🐠 Species: {label}")
        st.info(f"📊 Confidence: {conf:.2f}%")
