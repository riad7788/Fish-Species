import streamlit as st
import os
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np

# -----------------------
# Page Config
# -----------------------
st.set_page_config(
    page_title="Fish Species Detection",
    page_icon="🐟",
    layout="wide"
)

# -----------------------
# Sidebar Info
# -----------------------
st.sidebar.title("📌 Project Info")
st.sidebar.markdown("""
- **Course:** Capstone  
- **Method:** SimCLR (SSL)  
- **Framework:** PyTorch  
- **Web App:** Streamlit  
- **Developer:** Riad  
""")

# -----------------------
# Device
# -----------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -----------------------
# Model Definitions
# -----------------------
class SimCLR_Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        base_model = models.resnet18(weights=None)
        self.features = nn.Sequential(*list(base_model.children())[:-1])
        self.out_dim = 512

    def forward(self, x):
        x = self.features(x)
        return x.view(x.size(0), -1)

class Classifier(nn.Module):
    def __init__(self, in_dim, num_classes):
        super().__init__()
        self.fc = nn.Linear(in_dim, num_classes)

    def forward(self, x):
        return self.fc(x)

# -----------------------
# Class Names (Total 21)
# -----------------------
CLASS_NAMES = [
    "Biam", "Bata", "Batasio(tenra)","Chitul","Croaker(Poya)","Hilsha",
    "Kajoli","Meni","Pabda","Poli","Puti","Rita","Rui","Rupchanda",
    "Silver Carp","Telapiya","carp","Koi","kaikka","koral","shrimp"
]

# -----------------------
# Load Models
# -----------------------
@st.cache_resource
def load_models():
    # os.path.dirname ব্যবহার করার জন্য উপরে 'import os' করা হয়েছে
    base_dir = os.path.dirname(os.path.abspath(__file__))
    model_dir = os.path.join(base_dir, "models")

    # আপনার GitHub এর ফাইল নামের সাথে এগুলো মিল থাকতে হবে
    # যদি আপনার শুধু একটা .pt ফাইল থাকে, তবে এনকোডার লোড করার দরকার নেই যদি সেটা কম্বাইন্ড হয়।
    # আমি এখানে আপনার কোড অনুযায়ী পাথ সেট করছি:
    classifier_path = os.path.join(model_dir, "classifier.pt") # GitHub অনুযায়ী নাম

    if not os.path.exists(classifier_path):
        st.error(f"Model file not found at: {classifier_path}")
        st.stop()

    # এখানে num_classes অবশ্যই len(CLASS_NAMES) হতে হবে (২১)
    encoder = SimCLR_Encoder()
    classifier = Classifier(512, len(CLASS_NAMES))

    # লোড করা
    # দ্রষ্টব্য: যদি classifier.pt এর ভেতর এনকোডার ও ক্লাসিফায়ার একসাথে থাকে তবে কোড একটু বদলাতে হবে
    checkpoint = torch.load(classifier_path, map_location=device)
    classifier.load_state_dict(checkpoint) 

    encoder.to(device).eval()
    classifier.to(device).eval()

    return encoder, classifier

# মডেল কল করা
try:
    encoder, classifier = load_models()
except Exception as e:
    st.error(f"Error loading model: {e}")

# -----------------------
# Image Transform
# -----------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# -----------------------
# UI
# -----------------------
st.markdown("<h1 style='text-align:center;'>🐟 Fish Species Detection System</h1>", unsafe_allow_html=True)

uploaded_file = st.file_uploader("📤 Upload a fish image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)

    img_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        features = encoder(img_tensor)
        outputs = classifier(features)
        probs = torch.softmax(outputs, dim=1)

        pred_idx = torch.argmax(probs, dim=1).item()
        confidence = probs[0][pred_idx].item()

    st.success(f"🐠 **Predicted Species:** {CLASS_NAMES[pred_idx]}")
    st.info(f"🎯 **Confidence:** {confidence:.2%}")
