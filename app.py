import streamlit as st
import os
import requests
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import pandas as pd

# ==========================================
# ১. নোটবুক অনুযায়ী মডেল আর্কিটেকচার
# ==========================================
class SimCLR(nn.Module):
    def __init__(self, proj_dim=128):
        super().__init__()
        self.encoder = models.resnet50(weights=None)
        self.encoder.fc = nn.Identity() # নোটবুক অনুযায়ী
        self.projector = nn.Sequential(
            nn.Linear(2048, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, proj_dim)
        )
    def forward(self, x):
        h = self.encoder(x)
        return self.projector(h)

# ==========================================
# ২. ক্লাউড মডেল কনফিগ
# ==========================================
HF_EXPERT_URL = "https://huggingface.co/riad300/fish-simclr-encoder/resolve/main/fish_expert_weights.pt"
MODEL_PATH = "models/fish_expert_weights.pt"
os.makedirs("models", exist_ok=True)

st.set_page_config(page_title="Fish AI Pro", page_icon="🐟", layout="wide")

# ==========================================
# ৩. বর্ণানুক্রমিক সঠিক ক্লাস লিস্ট
# ==========================================
CLASS_NAMES = [
    "Baim", "Bata", "Batasio(tenra)", "Chitul", "Croaker(Poya)", 
    "Hilsha", "Kajoli", "Meni", "Pabda", "Poli", "Puti", 
    "Rita", "Rui", "Rupchada", "Silver Carp", "Telapiya", 
    "carp", "k", "kaikka", "koral", "shrimp"
]

# ==========================================
# ৪. হাই-প্রিসিশন ইঞ্জিন লোডার
# ==========================================
@st.cache_resource
def load_expert_engine():
    if not os.path.exists(MODEL_PATH):
        r = requests.get(HF_EXPERT_URL, stream=True)
        with open(MODEL_PATH, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192): f.write(chunk)
    
    try:
        # ক্লাসিফায়ার হিসেবে নোটবুকের আর্কিটেকচার
        base_model = SimCLR()
        # আপনার ট্রেনিং করা ২১টি ক্লাসের জন্য ফাইনাল লেয়ার
        classifier = nn.Sequential(
            base_model.encoder,
            nn.Linear(2048, 21)
        )
        sd = torch.load(MODEL_PATH, map_location=torch.device('cpu'))
        classifier.load_state_dict(sd, strict=True)
        classifier.eval()
        return classifier
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

expert_model = load_expert_engine()

# ==========================================
# ৫. ড্যাশবোর্ড ও লগইন সিস্টেম
# ==========================================
if 'user' not in st.session_state: st.session_state['user'] = None

with st.sidebar:
    st.title("🛡️ Access Control")
    if st.session_state['user']:
        st.success(f"Verified: {st.session_state['user']}")
        if st.button("Logout"): 
            st.session_state['user'] = None
            st.rerun()
    else:
        st.info("Please Login")

if not st.session_state['user']:
    st.markdown('<div style="background:rgba(255,255,255,0.1);padding:30px;border-radius:15px;"><h2>Expert Login</h2></div>', unsafe_allow_html=True)
    user = st.text_input("Username")
    if st.button("Login"):
        st.session_state['user'] = user
        st.rerun()
else:
    st.title("🐟 Fish Expert Analysis Dashboard")
    file = st.file_uploader("Upload Fish Specimen", type=["jpg", "png", "jpeg"])

    if file:
        img = Image.open(file).convert('RGB')
        col1, col2 = st.columns(2)
        with col1:
            st.image(img, caption="Target Specimen", use_container_width=True)
        
        with col2:
            if st.button("🚀 RUN ANALYSIS"):
                # নোটবুক অনুযায়ী ১৬০x১৬০ সাইজ
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
                
                st.success(f"Fish Identified: {CLASS_NAMES[idx.item()]}")
                st.metric("Confidence", f"{conf.item()*100:.2f}%")
                
                # টপ ৫ রেজাল্ট চার্ট
                top5_p, top5_i = torch.topk(prob, 5)
                df = pd.DataFrame({'Fish': [CLASS_NAMES[i] for i in top5_i], 'Confidence (%)': top5_p.numpy()*100})
                st.bar_chart(df, x='Fish', y='Confidence (%)')
