import streamlit as st
import os  # 1. os মডিউল যোগ করা হয়েছে
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
        # ResNet এর শেষ লেয়ার বাদ দিয়ে ফিচার এক্সট্রাক্টর তৈরি
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
    "Biam", "Bata", "Batasio(tenra)", "Chitul", "Croaker(Poya)", "Hilsha",
    "Kajoli", "Meni", "Pabda", "Poli", "Puti", "Rita", "Rui", "Rupchanda",
    "Silver Carp", "Telapiya", "carp", "Koi", "kaikka", "koral", "shrimp"
]

# -----------------------
# Load Models
# -----------------------
@st.cache_resource
def load_models():
    # ডিরেক্টরি পাথ সেট করা
    base_dir = os.path.dirname(os.path.abspath(__file__))
    model_dir = os.path.join(base_dir, "models")
    
    # GitHub অনুযায়ী ফাইলের নাম classifier.pt
    classifier_path = os.path.join(model_dir, "classifier.pt")

    if not os.path.exists(classifier_path):
        st.error(f"মডেল ফাইলটি পাওয়া যায়নি: {classifier_path}")
        st.stop()

    # মডেল ইনিশিয়ালাইজেশন (২১টি ক্লাসের জন্য)
    encoder = SimCLR_Encoder()
    classifier = Classifier(512, len(CLASS_NAMES))

    # 2. weights_only=False যোগ করা হয়েছে এরর এড়ানোর জন্য
    try:
        checkpoint = torch.load(classifier_path, map_location=device, weights_only=False)
        
        # যদি checkpoint সরাসরি state_dict হয়
        if isinstance(checkpoint, dict):
            classifier.load_state_dict(checkpoint)
        else:
            # যদি পুরো মডেল অবজেক্ট সেভ করা থাকে
            classifier = checkpoint
            
    except Exception as e:
        st.error(f"মডেল লোড করতে সমস্যা হয়েছে: {e}")
        st.stop()

    encoder.to(device).eval()
    classifier.to(device).eval()

    return encoder, classifier

# মডেল কল করা
encoder, classifier = load_models()

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
# UI Design
# -----------------------
st.markdown("<h1 style='text-align:center;'>🐟 Fish Species Detection System</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center;'>Self-Supervised Learning (SimCLR) based Fish Classification</p>", unsafe_allow_html=True)
st.write("---")

uploaded_file = st.file_uploader("📤 Upload a fish image (JPG, PNG, JPEG)", type=["jpg", "png", "jpeg"])

# -----------------------
# Prediction Logic
# -----------------------
if uploaded_file is not None:
    col1, col2 = st.columns([1, 1])
    
    image = Image.open(uploaded_file).convert("RGB")
    
    with col1:
        st.image(image, caption="Uploaded Image", use_container_width=True)

    # ইমেজ প্রসেসিং ও প্রেডিকশন
    img_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        features = encoder(img_tensor)
        outputs = classifier(features)
        probs = torch.softmax(outputs, dim=1)

        pred_idx = torch.argmax(probs, dim=1).item()
        confidence = probs[0][pred_idx].item()

    with col2:
        st.subheader("🔍 Prediction Result")
        st.success(f"🐠 **Predicted Species:** {CLASS_NAMES[pred_idx]}")
        st.info(f"🎯 **Confidence Level:** {confidence:.2%}")
        
        # প্রোগ্রেস বার দিয়ে কনফিডেন্স দেখানো
        st.progress(confidence)
