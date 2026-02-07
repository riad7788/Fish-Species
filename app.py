import streamlit as st
import os
import requests
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import pandas as pd

# ==========================================
# ১. নোটবুক সিঙ্কড আর্কিটেকচার (Cell-7)
# ==========================================
class SimCLR(nn.Module):
    def __init__(self, proj_dim=128):
        super().__init__()
        self.encoder = models.resnet50(weights=None)
        self.encoder.fc = nn.Identity() 
        self.projector = nn.Sequential(
            nn.Linear(2048, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, proj_dim)
        )
    def forward(self, x):
        return self.encoder(x) # আমরা শুধু এনকোডার পার্টটা ইউজ করছি

# ==========================================
# ২. রিসোর্স ও সঠিক ডিকশনারি ম্যাপিং
# ==========================================
HF_EXPERT_URL = "https://huggingface.co/riad300/fish-simclr-encoder/resolve/main/fish_expert_weights.pt"
MODEL_PATH = "models/fish_expert_weights.pt"
os.makedirs("models", exist_ok=True)

# এই ম্যাপিংটি আপনার ফোল্ডারের বর্ণানুক্রমিক সিরিয়াল (A-Z) অনুযায়ী ফিক্সড
FISH_CLASSES = {
    0: "Baim", 1: "Bata", 2: "Batasio(tenra)", 3: "Chitul", 4: "Croaker(Poya)",
    5: "Hilsha", 6: "Kajoli", 7: "Meni", 8: "Pabda", 9: "Poli", 10: "Puti",
    11: "Rita", 12: "Rui", 13: "Rupchada", 14: "Silver Carp", 15: "Telapiya",
    16: "carp", 17: "k", 18: "kaikka", 19: "koral", 20: "shrimp"
}

st.set_page_config(page_title="Fish AI - Final", page_icon="🐟", layout="wide")

# ==========================================
# ৩. ডার্ক মোড ইউআই ও ব্যাকগ্রাউন্ড
# ==========================================
def apply_theme():
    st.markdown("""
    <style>
    .stApp {
        background: linear-gradient(rgba(0,0,0,0.85), rgba(0,0,0,0.85)), 
                    url("https://images.unsplash.com/photo-1516734212186-a967f81ad0d7?q=80&w=2071") !important;
        background-size: cover !important;
        background-attachment: fixed !important;
    }
    .glass-card {
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(20px);
        border-radius: 20px; border: 1px solid rgba(0, 194, 255, 0.3);
        padding: 30px; color: white;
    }
    </style>
    """, unsafe_allow_html=True)

apply_theme()

# ==========================================
# ৪. পাওয়ারফুল ইঞ্জিন লোডার
# ==========================================
@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        r = requests.get(HF_EXPERT_URL, stream=True)
        with open(MODEL_PATH, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192): f.write(chunk)
    
    try:
        base = SimCLR()
        model = nn.Sequential(base.encoder, nn.Linear(2048, 21))
        sd = torch.load(MODEL_PATH, map_location=torch.device('cpu'))
        # কী-ক্লিনিং: যদি আপনার ওয়েটস ফাইলে 'encoder.' প্রিফিক্স থাকে
        new_sd = {k.replace("encoder.", "0.").replace("model.", "0."): v for k, v in sd.items()}
        model.load_state_dict(new_sd, strict=False)
        model.eval()
        return model
    except: return None

# ==========================================
# ৫. লগইন এবং অ্যাপ লজিক
# ==========================================
if 'logged_in' not in st.session_state: st.session_state['logged_in'] = False

if not st.session_state['logged_in']:
    st.markdown('<div class="glass-card"><h2>🛡️ Admin Access</h2></div>', unsafe_allow_html=True)
    user = st.text_input("Enter ID")
    if st.button("Unlock"):
        if user:
            st.session_state['logged_in'] = True
            st.rerun()
else:
    with st.sidebar:
        st.success("System Active")
        if st.button("Logout"):
            st.session_state['logged_in'] = False
            st.rerun()

    st.markdown('<div class="glass-card"><h1>🐟 Fish Expert System</h1></div>', unsafe_allow_html=True)
    
    file = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])
    
    if file:
        img = Image.open(file).convert('RGB')
        col1, col2 = st.columns(2)
        
        with col1:
            st.image(img, use_container_width=True, caption="Analyzed Image")
        
        with col2:
            if st.button("🚀 PREDICT SPECIES"):
                model = load_model()
                if model:
                    # নোটবুক অনুযায়ী ১৬০x১৬০ সাইজ (Cell-2)
                    transform = transforms.Compose([
                        transforms.Resize((160, 160)),
                        transforms.ToTensor(),
                        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                    ])
                    
                    tensor = transform(img).unsqueeze(0)
                    
                    with torch.no_grad():
                        output = model(tensor)
                        prob = torch.nn.functional.softmax(output[0], dim=0)
                        conf, idx = torch.max(prob, 0)
                    
                    # সঠিক ডিকশনারি থেকে নাম নেওয়া
                    fish_name = FISH_CLASSES.get(idx.item(), "Unknown")
                    
                    st.markdown(f"""
                        <div style="background:rgba(0,194,255,0.1); border:1px solid #00C2FF; padding:20px; border-radius:15px;">
                            <h2 style="color:#00C2FF;">Result: {fish_name}</h2>
                            <h3>Confidence: {conf.item()*100:.2f}%</h3>
                        </div>
                    """, unsafe_allow_html=True)
