import streamlit as st
import torch
from torchvision import transforms
from PIL import Image

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

CLASS_NAMES = [
    "Baim","Bata","Batasio(tenra)","Chitul","Croaker(Poya)",
    "Hilsha","Kajoli","Meni","Pabda","Poli","Puti",
    "Rita","Rui","Rupchada","Silver Carp","Telapiya",
    "carp","k","kaikka","koral","shrimp"
]

@st.cache_resource
def load_models():
    encoder = torch.load("models/encoder_simclr.pt", map_location=DEVICE)
    classifier = torch.load("models/classifier_final.pt", map_location=DEVICE)

    encoder.eval()
    classifier.eval()
    return encoder, classifier

encoder, classifier = load_models()

transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])

def predict(img):
    img = transform(img).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        feat = encoder(img)
        out = classifier(feat)
        prob = torch.softmax(out, dim=1)
        idx = prob.argmax(1).item()
    return CLASS_NAMES[idx], prob[0][idx].item()*100

st.set_page_config("Fish Species Detection")

st.title("🐟 Fish Species Detection & Classification")

file = st.file_uploader("Upload Fish Image", ["jpg","png","jpeg"])

if file:
    img = Image.open(file).convert("RGB")
    st.image(img, use_column_width=True)

    if st.button("Predict"):
        label, conf = predict(img)
        st.success(f"🐠 Species: {label}")
        st.info(f"Confidence: {conf:.2f}%")
