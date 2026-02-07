import streamlit as st
import os
import requests
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import pandas as pd

# ==========================================
# 1. CORE CONFIG & EXPERT WEIGHTS
# ==========================================
HF_EXPERT_URL = "https://huggingface.co/riad300/fish-simclr-encoder/resolve/main/fish_expert_weights.pt"
MODEL_PATH = "models/fish_expert_weights.pt"
os.makedirs("models", exist_ok=True)

st.set_page_config(page_title="Fish AI Pro", page_icon="🐟", layout="wide")

# ==========================================
# 2. UI THEME & BACKGROUND FIX (১০০% কাজ করবে)
# ==========================================
def apply_pro_styling():
    # এখানে ডিরেক্ট ইমেজ ইউআরএল দেওয়া হয়েছে যেন ব্যাকগ্রাউন্ড মিস না হয়
    st.markdown("""
    <style>
    .stApp {
        background: linear-gradient(rgba(0,0,0,0.85), rgba(0,0,0,0.85)), 
                    url("https://images.unsplash.com/photo-1516734212186-a967f81ad0d7?q=80&w=2071") !important;
        background-size: cover !important;
        background-attachment: fixed !important;
        color: #ffffff;
    }
    .glass-card {
        background: rgba(255, 255, 255, 0.07);
        backdrop-filter: blur(15px);
        border-radius: 20px;
        border: 1px solid rgba(0, 194, 255, 0.3);
        padding: 30px;
        margin-bottom: 20px;
    }
    div.stButton > button {
        background: linear-gradient(90deg, #00C2FF, #0072FF);
        color: white; border-radius: 12px; height: 3.5em; font-weight: bold; width: 100%; border: none;
    }
    </style>
    """, unsafe_allow_html=True)

apply_pro_styling()

# ==========================================
# 3. HIGH-PRECISION MODEL LOADER
# ==========================================
@st.cache_resource
def load_expert_engine():
    if not os.path.exists(MODEL_PATH):
        try:
            r = requests.get(HF_EXPERT_URL, stream=True)
            with open(MODEL_PATH, "wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
        except: return None, "Cloud Connection Error"
    
    try:
        # ResNet50 Architecture
        model = models.resnet50(weights=None)
        model.fc = nn.Linear(model.fc.in_features, 21)
        
        # State Dict Loading with Extra Safety
        checkpoint = torch.load(MODEL_PATH, map_location=torch.device('cpu'))
        
        # যদি ফাইলটি শুধু weights হয় তবে একরকম, যদি পুরো ডিকশনারি হয় তবে আরেকরকম
        if 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint
            
        # SimCLR Key Mapping (Clean all prefixes)
        clean_sd = {}
        for k, v in state_dict.items():
            name = k.replace("encoder.", "").replace("model.", "").replace("backbone.", "").replace("module.", "")
            clean_sd[name] = v
            
        model.load_state_dict(clean_sd, strict=False)
        model.eval()
        return model, "Expert Engine Operational"
    except Exception as e:
        return None, f"Loading Error: {str(e)}"

expert_model, engine_status = load_expert_engine()

CLASS_NAMES = [
    "Baim", "Bata", "Batasio(tenra)", "Chitul", "Croaker(Poya)", 
    "Hilsha", "Kajoli", "Meni", "Pabda", "Poli", "Puti", 
    "Rita", "Rui", "Rupchada", "Silver Carp", "Telapiya", 
    "carp", "k", "kaikka", "koral", "shrimp"
]

# ==========================================
# 4. ANALYZER INTERFACE
# ==========================================
if 'user' not in st.session_state: st.session_state['user'] = "Guest Expert"

with st.sidebar:
    st.title("🛡️ System Control")
    st.info(f"User: {st.session_state['user']}")
    st.success(f"Status: {engine_status}")
    st.write("Industry Grade Build 4.0")

st.markdown('<div class="glass-card"><h1>🐟 Fish Identification Analytics</h1><p>Hugging Face Expert Weights Integrated</p></div>', unsafe_allow_html=True)

source = st.file_uploader("Drop Specimen Image Here", type=["jpg", "png", "jpeg"])

if source:
    col1, col2 = st.columns([1, 1])
    with col1:
        image = Image.open(source).convert('RGB')
        st.image(image, caption="Analyzed Specimen", use_container_width=True)
    
    with col2:
        if st.button("RUN DEEP ANALYSIS"):
            if expert_model:
                with st.spinner("Decoding Neural Patterns..."):
                    # INDUSTRY STANDARD PREPROCESSING (SimCLR specific)
                    transform = transforms.Compose([
                        transforms.Resize(256),
                        transforms.CenterCrop(224),
                        transforms.ToTensor(),
                        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                    ])
                    
                    tensor = transform(image).unsqueeze(0)
                    
                    with torch.no_grad():
                        logits = expert_model(tensor)
                        probs = torch.nn.functional.softmax(logits[0], dim=0)
                        conf, idx = torch.max(probs, 0)
                    
                    # Result Display
                    st.markdown(f'''
                        <div style="background: rgba(0, 194, 255, 0.1); border: 2px solid #00C2FF; padding: 20px; border-radius: 15px;">
                            <h2 style="color: #00C2FF; margin:0;">Specimen: {CLASS_NAMES[idx.item()]}</h2>
                            <h3 style="margin:0;">Confidence: {conf.item()*100:.2f}%</h3>
                        </div>
                    ''', unsafe_allow_html=True)
                    
                    # Probability Graph
                    top5_p, top5_i = torch.topk(probs, 5)
                    chart_data = pd.DataFrame({
                        'Species': [CLASS_NAMES[i] for i in top5_i],
                        'Probability (%)': top5_p.numpy() * 100
                    })
                    st.write("#### Confidence Distribution")
                    st.bar_chart(chart_data, x='Species', y='Probability (%)', horizontal=True)

st.markdown('<p style="text-align:center; color:rgba(255,255,255,0.3); margin-top:50px;">© 2026 RIAD AI INDUSTRIES | ENTERPRISE DEPLOYMENT</p>', unsafe_allow_html=True)
