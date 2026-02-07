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

# --------------------------------------------------
# LOAD MODELS
# --------------------------------------------------
@st.cache_resource
def load_models():

    # 🔽 Download encoder state_dict from Hugging Face
    encoder_path = hf_hub_download(
        repo_id="riad300/fish-simclr-encoder",
        filename="encoder_simclr.pt"
    )

    raw_state = torch.load(encoder_path, map_location=DEVICE)

    # 🔽 Rebuild encoder architecture (ResNet50 backbone)
    base = models.resnet50(weights=None)
    encoder = nn.Sequential(*list(base.children())[:-1])
    encoder.to(DEVICE)

    # 🔽 FIX SimCLR key mismatch
    cleaned_state = {}
    for k, v in raw_state.items():
        k = k.replace("encoder.", "")
        k = k.replace("backbone.", "")
        k = k.replace("module.", "")
        cleaned_state[k] = v

    encoder.load_state_dict(cleaned_state, strict=False)
    encoder.eval()

    # 🔽 Load classifier (local small file)
    classifier = torch.load(
        "models/classifier_final.pt",
        map_location=DEVICE
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
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# --------------------------------------------------
# PREDICTION FUNCTION
# --------------------------------------------------
def predict(img):
    img = transform(img).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        feat = encoder(img)
        feat = feat.view(feat.size(0), -1)   # 🔥 VERY IMPORTANT
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

uploaded_file = st.file_uploader(
    "📤 Upload Fish Image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    if st.button("🔍 Predict"):
        with st.spinner("Analyzing image..."):
            label, confidence = predict(image)

        st.success(f"🐠 **Predicted Species:** {label}")
        st.info(f"📊 **Confidence:** {confidence:.2f}%")
