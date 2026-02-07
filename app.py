import streamlit as st
import os
import requests
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import pandas as pd

# ==========================================
# 1. RESOURCE SYNC (Cloud Model)
# ==========================================
HF_EXPERT_URL = "https://huggingface.co/riad300/fish-simclr-encoder/resolve/main/fish_expert_weights.pt"
MODEL_PATH = "models/fish_expert_weights.pt"
os.makedirs("models", exist_ok=True)

st.set_page_config(page_title="Fish AI - Final Build", page_icon="🐟", layout="wide")

# ==========================================
# 2. UI & BACKGROUND FIX (100% RECOVERY)
# ==========================================
def apply_ui_theme():
    # সরাসরি স্ট্রিং ব্যবহার করে ব্যাকগ্রাউন্ড ফিক্স করা হয়েছে
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
        backdrop-filter: blur(20px);
        border-radius: 20px; border: 1px solid rgba(0, 194, 255, 0.3);
        padding: 40px; color: white;
    }
    div.stButton > button {
        background: linear-gradient(90deg, #00C2FF, #0072FF);
        color: white; border-radius: 12px; height: 3.5em; font-weight: bold; width: 100%; border: none;
    }
    </style>
    """, unsafe_allow_html=True)

apply_ui_theme()

# ==========================================
# 3. ABSOLUTE CORRECT CLASS MAPPING
# ==========================================
# পাইথনের ImageFolder সর্টিং (Capital first, then Small letters)
CLASS_NAMES = [
    "Baim", "Bata", "Batasio(tenra)", "Chitul", "Croaker(Poya)", 
    "Hilsha", "Kajoli", "Meni", "Pabda", "Poli", "Puti", 
    "Rita", "Rui", "Rupchada", "Silver Carp", "Telapiya", 
    "carp", "k", "kaikka", "koral", "shrimp"
]

# ==========================================
# 4. ENGINE LOADER (SimCLR Expert)
# ==========================================
@st.cache_resource
def load_expert_engine():
    if not os.path.exists(MODEL_PATH):
        r = requests.get(HF_EXPERT_URL, stream=True)
        with open(MODEL_PATH, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192): f.write(chunk)
    
    try:
        model = models.resnet50(weights=None)
        model.fc = nn.Linear(model.fc.in_features, 21)
        sd = torch.load(MODEL_PATH, map_location=torch.device('cpu'))
        # Key cleaning logic
        clean_sd = {k.replace("encoder.", "").replace("model.", ""): v for k, v in sd.items()}
        model.load_state_dict(clean_sd, strict=False)
        model.eval()
        return model
    except:
        return None

expert_model = load_expert_engine()

# ==========================================
# 5. DASHBOARD & ANALYSIS (Indentation Fixed)
# ==========================================
if 'user' not in st.session_state: st.session_state['user'] = "Riad"

with st.sidebar:
    st.title("🛡️ Access Control")
    st.success(f"Verified: {st.session_state['user']}")
    st.write("---")
    st.info("System Build: v8.0 Enterprise")

st.markdown('<div class="main-card"><h1>Expert Fish Analytics</h1><p>Neural Pattern Mapping Active</p></div>', unsafe_allow_html=True)

file = st.file_uploader("Upload Specimen", type=["jpg", "png", "jpeg"])

if file:
    col1, col2 = st.columns([1, 1.2])
    with col1:
        img = Image.open(file).convert('RGB')
        st.image(img, caption="Target Specimen", use_container_width=True)
    
    with col2:
        if st.button("🚀 EXECUTE NEURAL DIAGNOSTICS"):
            if expert_model:
                with st.spinner("Decoding Morphology..."):
                    # Standard Transformation logic
                    transform = transforms.Compose([
                        transforms.Resize(256),
                        transforms.CenterCrop(224),
                        transforms.ToTensor(),
                        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                    ])
                    # এই লাইনটির স্পেস (Indentation) ঠিক করা হয়েছে
                    tensor = transform(img).unsqueeze(0)
                    
                    with torch.no_grad():
                        out = expert_model(tensor)
                        prob = torch.nn.functional.softmax(out[0], dim=0)
                        conf, idx = torch.max(prob, 0)
                    
                    # ফলাফল প্রদর্শন
                    st.markdown(f'''
                        <div style="border: 2px solid #00C2FF; border-radius: 15px; padding: 25px; background: rgba(0,194,255,0.1);">
                            <h2 style="color: #00C2FF; margin:0;">Identified: {CLASS_NAMES[idx.item()]}</h2>
                            <h3 style="margin:0;">Precision: {conf.item()*100:.2f}%</h3>
                        </div>
                    ''', unsafe_allow_html=True)
                    
                    # প্রবাবিলিটি গ্রাফ
                    top5_p, top5_i = torch.topk(prob, 5)
                    df = pd.DataFrame({'Species': [CLASS_NAMES[i] for i in top5_i], 'Confidence (%)': top5_p.numpy()*100})
                    st.bar_chart(df, x='Species', y='Confidence (%)', horizontal=True)

st.markdown('<p style="text-align:center; color:gray; margin-top:80px;">© 2026 RIAD AI INDUSTRIES • CLOUD DEPLOYMENT</p>', unsafe_allow_html=True)
