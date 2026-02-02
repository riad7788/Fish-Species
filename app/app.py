import streamlit as st
import torch
from PIL import Image
import torchvision.transforms as transforms

# ----------------------------
# Page Config
# ----------------------------
st.set_page_config(
    page_title="🐟 Fish Species Classifier",
    page_icon="🐠",
    layout="wide",
)

# ----------------------------
# Sidebar
# ----------------------------
st.sidebar.title("Settings")
confidence_threshold = st.sidebar.slider("Confidence threshold", 0.0, 1.0, 0.5)

with st.sidebar.expander("About"):
    st.write("""
    - SimCLR encoder + linear classifier  
    - Upload a fish image to predict species  
    - Professional Streamlit UI
    """)

# ----------------------------
# Class names
# ----------------------------
class_names = ["Biam", "Bata", "Batasio(tenra)","Chitul","Croaker(Poya)","Hilsha","Kajoli","Meni","Pabda","Poli","Puti","Rita","Rui","Rupchanda","Silver Carp","Telapiya","carp","Koi","kaikka","koral","shrimp"]

# ----------------------------
# Load Models (cached)
# ----------------------------
@st.cache_resource
def load_models():
    encoder = torch.load("models/encoder.pt", map_location="cpu")
    encoder.eval()
    classifier = torch.load("models/classifier.pt", map_location="cpu")
    classifier.eval()
    return encoder, classifier

encoder, classifier = load_models()
st.success("✅ Models loaded successfully!")

# ----------------------------
# File uploader
# ----------------------------
uploaded_file = st.file_uploader("Upload Fish Image", type=["jpg","png","jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # ----------------------------
    # Preprocess image
    # ----------------------------
    transform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
    ])
    img_tensor = transform(image).unsqueeze(0)  # batch size 1

    # ----------------------------
    # Buttons
    # ----------------------------
    col1, col2 = st.columns(2)
    classify_btn = col1.button("🟢 Classify")
    clear_btn = col2.button("🔴 Clear")

    if classify_btn:
        with torch.no_grad():
            features = encoder(img_tensor)
            output = classifier(features)
            probs = torch.softmax(output, dim=1).numpy()[0]
            class_idx = probs.argmax()
            confidence = probs[class_idx]

        if confidence >= confidence_threshold:
            st.subheader("Prediction Result")
            st.success(f"Predicted: {class_names[class_idx]} (Confidence: {confidence:.2f})")
        else:
            st.warning("Confidence too low. Prediction uncertain.")

    if clear_btn:
        st.experimental_rerun()
