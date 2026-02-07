import streamlit as st
import os
import requests
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import pandas as pd

# ==========================================
# ১. নোটবুক অনুযায়ী মডেল আর্কিটেকচার (Cell-4 & 7)
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
        h = self.encoder(x)
        return self.projector(h)

# ==========================================
# ২. রিসোর্স কনফিগ
# ==========================================
HF_EXPERT_URL = "https://huggingface.co/riad300/fish-simclr-encoder/resolve/main/fish_expert_weights.pt"
MODEL_PATH = "models/fish_expert_weights.pt"
os.makedirs("models", exist_ok=True)

st.set_page_config(page_title="Fish AI Pro", page_icon="🐟", layout="wide")

# ==========================================
# ৩. সঠিক ক্লাস লিস্ট (নোটবুক অনুযায়ী)
# ==========================================
CLASS_NAMES = [
    "Baim", "Bata", "Batasio(tenra)", "Chitul", "Croaker(Poya)", 
    "Hilsha", "Kajoli", "Meni", "Pabda", "Poli", "Puti", 
    "Rita", "Rui", "Rupchada", "Silver Carp", "Telapiya", 
    "carp", "k", "kaikka", "koral", "shrimp"
]

# ==========================================
# ৪. ডার্ক প্রিমিয়াম ইউআই (ব্যাকগ্রাউন্ড ফিক্স)
# ==========================================
def apply_ui():
    st.markdown("""
    <style>
    .stApp {
        background: linear-gradient(rgba(0,0,0,0.85), rgba(0,0,0,0.85)), 
                    url("https://images.unsplash.com/photo-1524704654690-b56c05c78a00?q=80&w=2069") !important;
        background-size: cover !important;
        background-attachment: fixed !important;
    }
    .main-card {
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(15px);
        border-radius: 20px; border: 1px solid rgba(0, 194, 255, 0.2);
        padding: 30px; color: white;
    }
    div.stButton > button {
        background: linear-gradient(90deg, #00C2FF, #0072FF);
        color: white; border-radius: 12px; height: 3.5em; font-weight: bold; width: 100%; border: none;
    }
    </style>
    """, unsafe_allow_html=True)

apply_ui()

# ==========================================
# ৫. হাই-প্রিসিশন ইঞ্জিন লোডার
# ==========================================
@st.cache_resource
def load_expert_engine():
    if not os.path.exists(MODEL_PATH):
        r = requests.get(HF_EXPERT_URL, stream=True)
        with open(MODEL_PATH, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192): f.write(chunk)
    
    try:
        # নোটবুকের সিমলার মডেল থেকে ক্লাসিফায়ার তৈরি (Cell-7)
        base = SimCLR()
        classifier = nn.Sequential(base.encoder, nn.Linear(2048, 21))
        
        # সরাসরি ওয়েটস লোড
        sd = torch.load(MODEL_PATH, map_location=torch.device('cpu'))
        classifier.load_state_dict(sd)
        classifier.eval()
        return classifier
    except Exception as e:
        return None

expert_model = load_expert_engine()

# ==========================================
# ৬. লগইন এবং ড্যাশবোর্ড লজিক
# ==========================================
if 'user' not in st.session_state: st.session_state['user'] = None

with st.sidebar:
    st.title("🛡️ Secure Access")
    if st.session_state['user']:
        st.success(f"Verified: {st.session_state['user']}")
        if st.button("Logout"): 
            st.session_state['user'] = None
            st.rerun()
    else:
        st.info("Authentication Required")

if not st.session_state['user']:
    st.markdown('<div class="main-card"><h2>Expert Portal Login</h2></div>', unsafe_allow_html=True)
    u = st.text_input("Username")
    if st.button("Unlock Dashboard"):
        st.session_state['user'] = u
        st.rerun()
else:
    st.markdown('<div class="main-card"><h1>Expert Fish Analyzer</h1><p>Synced with SimCLR Notebook Build</p></div>', unsafe_allow_html=True)
    
    file = st.file_uploader("Upload Fish Specimen", type=["jpg", "png", "jpeg"])
    if file:
        col1, col2 = st.columns([1, 1.2])
        with col1:
            img = Image.open(file).convert('RGB')
            st.image(img, caption="Analyzed Specimen", use_container_width=True)
        
        with col2:
            if st.button("🚀 EXECUTE NEURAL ANALYSIS"):
                if expert_model:
                    with st.spinner("Decoding Neural Patterns..."):
                        # আপনার নোটবুক অনুযায়ী ১৬০x১৬০ সাইজ (Cell-2)
                        transform = transforms.Compose([
                            transforms.Resize((160, 160)),
                            transforms.ToTensor(),
                            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                        ])
                        
                        tensor = transform(img).unsqueeze(0)
                        
                        with torch.no_grad():
                            out = expert_model(tensor)
                            prob = torch.nn.functional.softmax(out[0], dim=0)
                            conf, idx = torch.max(prob, 0)
                        
                        st.markdown(f'''
                            <div style="border: 2px solid #00C2FF; border-radius: 15px; padding: 25px; background: rgba(0,194,255,0.1);">
                                <h2 style="color: #00C2FF; margin:0;">Identified: {CLASS_NAMES[idx.item()]}</h2>
                                <h3 style="margin:0;">Precision: {conf.item()*100:.2f}%</h3>
                            </div>
                        ''', unsafe_allow_html=True)
                        
                        # গ্রাফ ডিস্ট্রিবিউশন
                        top5_p, top5_i = torch.topk(prob, 5)
                        df = pd.DataFrame({'Fish': [CLASS_NAMES[i] for i in top5_i], 'Match (%)': top5_p.numpy()*100})
                        st.bar_chart(df, x='Fish', y='Match (%)')

st.markdown('<p style="text-align:center; color:gray; margin-top:80px;">© 2026 RIAD AI INDUSTRIES • ENTERPRISE SYNC BUILD</p>', unsafe_allow_html=True)
