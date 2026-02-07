import streamlit as st
import os
import requests
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import pandas as pd

# ==========================================
# 1. EXPERT RESOURCE CONFIG
# ==========================================
HF_EXPERT_URL = "https://huggingface.co/riad300/fish-simclr-encoder/resolve/main/fish_expert_weights.pt"
MODEL_PATH = "models/fish_expert_weights.pt"
os.makedirs("models", exist_ok=True)

st.set_page_config(page_title="Fish AI Expert", page_icon="🐟", layout="wide")

# ==========================================
# 2. UI & BACKGROUND RESTORATION
# ==========================================
def apply_ui():
    # ব্যাকগ্রাউন্ড ইমেজ ফিরিয়ে আনা হয়েছে
    st.markdown("""
    <style>
    .stApp {
        background: linear-gradient(rgba(0,0,0,0.8), rgba(0,0,0,0.8)), 
                    url("https://images.unsplash.com/photo-1524704654690-b56c05c78a00?q=80&w=2069");
        background-size: cover !important;
        background-attachment: fixed;
    }
    .glass-card {
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(15px);
        border-radius: 20px; border: 1px solid rgba(0, 194, 255, 0.2);
        padding: 30px; margin-bottom: 20px; color: white;
    }
    </style>
    """, unsafe_allow_html=True)

apply_ui()

# ==========================================
# 3. CORRECT CLASS MAPPING (Verified)
# ==========================================
# আপনার ফোল্ডার স্ট্রাকচার অনুযায়ী সঠিক সিরিয়াল
CLASS_NAMES = [
    "Baim", "Bata", "Batasio(tenra)", "Chitul", "Croaker(Poya)", 
    "Hilsha", "Kajoli", "Meni", "Pabda", "Poli", "Puti", 
    "Rita", "Rui", "Rupchada", "Silver Carp", "Telapiya", 
    "carp", "k", "kaikka", "koral", "shrimp"
]

# ==========================================
# 4. EXPERT ENGINE LOADER
# ==========================================
@st.cache_resource
def load_engine():
    if not os.path.exists(MODEL_PATH):
        r = requests.get(HF_EXPERT_URL, stream=True)
        with open(MODEL_PATH, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192): f.write(chunk)
    
    try:
        model = models.resnet50(weights=None)
        model.fc = nn.Linear(model.fc.in_features, 21)
        checkpoint = torch.load(MODEL_PATH, map_location=torch.device('cpu'))
        
        # State Dict Clean-up
        sd = checkpoint.get('state_dict', checkpoint)
        clean_sd = {k.replace("encoder.", "").replace("model.", ""): v for k, v in sd.items()}
        
        model.load_state_dict(clean_sd, strict=False)
        model.eval()
        return model
    except Exception as e: return str(e)

expert_model = load_engine()

# ==========================================
# 5. AUTH & NAVIGATION
# ==========================================
if 'user' not in st.session_state: st.session_state['user'] = None

with st.sidebar:
    st.title("🛡️ System Control")
    if st.session_state['user']:
        st.success(f"User: {st.session_state['user']}")
        nav = st.radio("Menu", ["Dashboard", "Logout"])
    else:
        nav = st.radio("Menu", ["Login"])
    st.write("---")
    st.write("Industry Grade Build 4.5")

# ==========================================
# 6. APP LOGIC
# ==========================================
if nav == "Login":
    st.markdown('<div class="glass-card"><h2>Expert Portal</h2></div>', unsafe_allow_html=True)
    u = st.text_input("Username")
    if st.button("Enter"):
        st.session_state['user'] = u
        st.rerun()

elif nav == "Logout":
    st.session_state['user'] = None
    st.rerun()

elif nav == "Dashboard":
    st.markdown('<div class="glass-card"><h1>Fish Species Detection</h1></div>', unsafe_allow_html=True)
    
    file = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])
    if file:
        col1, col2 = st.columns(2)
        with col1:
            img = Image.open(file).convert('RGB')
            st.image(img, caption="Analyzed Specimen", use_container_width=True)
        
        with col2:
            if st.button("RUN AI ANALYSIS"):
                # Professional Preprocessing
                transform = transforms.Compose([
                    transforms.Resize(256),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                ])
                tensor = transform(img).unsqueeze(0)
                
                with torch.no_grad():
                    output = expert_model(tensor)
                    prob = torch.nn.functional.softmax(output[0], dim=0)
                    conf, idx = torch.max(prob, 0)
                
                # রেজাল্ট নির্ভুল করার ফিল্টার
                if conf.item() < 0.60:
                    st.warning("⚠️ Low Confidence. Result might be inaccurate.")

                st.markdown(f'''
                    <div style="border: 2px solid #00C2FF; padding: 20px; border-radius: 15px; background: rgba(0,194,255,0.1);">
                        <h2 style="color: #00C2FF;">Specimen: {CLASS_NAMES[idx.item()]}</h2>
                        <h3>Confidence: {conf.item()*100:.2f}%</h3>
                    </div>
                ''', unsafe_allow_html=True)
                
                # Probability Distribution Graph
                top5_p, top5_i = torch.topk(prob, 5)
                chart = pd.DataFrame({'Species': [CLASS_NAMES[i] for i in top5_i], 'Confidence (%)': top5_p.numpy()*100})
                st.bar_chart(chart, x='Species', y='Confidence (%)', horizontal=True)
