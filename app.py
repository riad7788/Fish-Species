import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import os

# --- ১. মডেল স্ট্রাকচার (আপনার নোটবুক অনুযায়ী ResNet50) ---
class SimCLR_Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        # নোটবুক অনুযায়ী resnet50 ব্যবহার করা হয়েছে
        base_model = models.resnet50(weights=None)
        # লাস্ট লেয়ার (fc) বাদ দিয়ে ফিচার এক্সট্রাক্টর তৈরি
        self.encoder = nn.Sequential(*list(base_model.children())[:-1])

    def forward(self, x):
        h = self.encoder(x)
        return h.view(h.size(0), -1) # আউটপুট ডাইমেনশন: 2048

class Classifier(nn.Module):
    def __init__(self, in_dim, num_classes):
        super().__init__()
        self.fc = nn.Linear(in_dim, num_classes)

    def forward(self, x):
        return self.fc(x)

# --- ২. মডেল লোডিং ফাংশন ---
@st.cache_resource
def load_full_system():
    device = torch.device("cpu")
    
    # এনকোডার ডাউনলোড ও লোড (আপনার Hugging Face লিঙ্ক)
    ENCODER_URL = "https://huggingface.co/riad300/fish-simclr-encoder/resolve/main/encoder_simclr.pt"
    
    encoder = SimCLR_Encoder()
    try:
        # সরাসরি লিঙ্ক থেকে এনকোডার লোড করা
        state_dict = torch.hub.load_state_dict_from_url(ENCODER_URL, map_location=device)
        encoder.load_state_dict(state_dict)
    except Exception as e:
        st.error(f"Encoder loading error: {e}")

    # ক্লাসিফায়ার লোড (এটি GitHub এর models/ ফোল্ডার থেকে আসবে)
    base_dir = os.path.dirname(os.path.abspath(__file__))
    classifier_path = os.path.join(base_dir, "models", "classifier.pt")
    
    # ResNet50 এর জন্য ইনপুট ২০৪৮ এবং মাছ ২১টি
    classifier = Classifier(2048, 21) 
    
    if os.path.exists(classifier_path):
        try:
            # আপনি state_dict সেভ করেছেন, তাই এটি লোড হবে
            c_state = torch.load(classifier_path, map_location=device, weights_only=False)
            classifier.load_state_dict(c_state)
        except Exception as e:
            st.error(f"Classifier weights loading error: {e}")
    else:
        st.warning("Classifier weights not found! Please upload 'classifier.pt' to models/ folder.")
        
    encoder.eval()
    classifier.eval()
    return encoder, classifier

# --- ৩. ইউজার ইন্টারফেস (UI) ---
st.set_page_config(page_title="Fish AI Expert", page_icon="🐟", layout="centered")

st.markdown("<h1 style='text-align: center; color: #1E88E5;'>🐟 Fish Species AI Classifier</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Self-Supervised Learning (SimCLR) with ResNet50</p>", unsafe_allow_html=True)
st.write("---")

# মডেল কল করা
with st.spinner('AI Models are loading... Please wait.'):
    encoder, classifier = load_full_system()

# আপনার নোটবুকের ২১টি মাছের নাম
CLASSES = [
    "Biam", "Bata", "Batasio(tenra)", "Chitul", "Croaker(Poya)", "Hilsha",
    "Kajoli", "Meni", "Pabda", "Poli", "Puti", "Rita", "Rui", "Rupchanda",
    "Silver Carp", "Telapiya", "carp", "Koi", "kaikka", "koral", "shrimp"
]

uploaded_file = st.file_uploader("📤 একটি মাছের ছবি আপলোড করুন...", type=["jpg", "png", "jpeg"])

if uploaded_file:
    img = Image.open(uploaded_file).convert("RGB")
    
    col1, col2 = st.columns([1, 1])
    with col1:
        st.image(img, caption="Uploaded Image", use_container_width=True)
    
    # ইমেজ প্রসেসিং
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    input_tensor = preprocess(img).unsqueeze(0)
    
    with torch.no_grad():
        features = encoder(input_tensor)
        outputs = classifier(features)
        probs = torch.softmax(outputs, dim=1)
        confidence, idx = torch.max(probs, 1)
    
    with col2:
        st.subheader("🔍 Result")
        st.success(f"**Species:** {CLASSES[idx.item()]}")
        st.info(f"**Confidence:** {confidence.item():.2%}")
        st.progress(confidence.item())

st.write("---")
st.caption("Developed by Riad | Fish Species Detection System")
