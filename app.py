import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
from huggingface_hub import hf_hub_download

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
FEATURE_DIM = 2048   # ResNet50 output

# --------------------------------------------------
# LOAD MODELS
# --------------------------------------------------
@st.cache_resource
def load_models():

    # ================= ENCODER =================
    encoder_path = hf_hub_download(
        repo_id="riad300/fish-simclr-encoder",
        filename="encoder_simclr.pt"
    )

    encoder_state = torch.load(encoder_path, map_location=DEVICE)

    base = models.resnet50(weights=None)
    encoder = nn.Sequential(*list(base.children())[:-1])
    encoder.to(DEVICE)

    # clean SimCLR keys
    clean_state = {}
    for k, v in encoder_state.items():
        k = k.replace("encoder.", "")
        k = k.replace("backbone.", "")
        k = k.replace("module.", "")
        clean_state[k] = v

    encoder.load_state_dict(clean_state, strict=False)
    encoder.eval()

    # ================= CLASSIFIER =================
    classifier_state = torch.load(
        "models/classifier_final.pt",
        map_location=DEVICE
    )

    classifier = nn.Linear(FEATURE_DIM, NUM_CLASSES)
    classifier.load_state_dict(classifier_state)
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
# PREDICT
# --------------------------------------------------
def predict(img):
    img = transform(img).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        feat = encoder(img)
        feat = feat.view(feat.size(0), -1)   # flatten
        out = classifier(feat)
        prob = torch.softmax(out, dim=1)

    idx = prob.argmax(1).item()
    return CLASS_NAMES[idx], prob[0][idx].item() * 100

# --------------------------------------------------
# STREAMLIT UI
# --------------------------------------------------
st.set_page_config(
    page_title="Fish Species Detection",
    layout="centered"
)

st.title("🐟 Fish Species Detection & Classification")
st.write("Upload a fish image and the AI model will predict the species.")

file = st.file_uploader(
    "📤 Upload Fish Image",
    type=["jpg", "jpeg", "png"]
)

if file:
    image = Image.open(file).convert("RGB")
    st.image(image, use_column_width=True)

    if st.button("🔍 Predict"):
        with st.spinner("Analyzing image..."):
            label, conf = predict(image)

        st.success(f"🐠 **Predicted Species:** {label}")
        st.info(f"📊 **Confidence:** {conf:.2f}%")
