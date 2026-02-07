import streamlit as st
import os
import requests
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import pandas as pd
from werkzeug.security import generate_password_hash, check_password_hash

# ==========================================
# 1. CLOUD MODEL CONFIGURATION
# ==========================================
# সরাসরি আপনার দেওয়া Expert Weights লিঙ্ক ব্যবহার করা হচ্ছে
HF_EXPERT_URL = "https://huggingface.co/riad300/fish-simclr-encoder/resolve/main/fish_expert_weights.pt"
MODEL_PATH = "models/fish_expert_weights.pt"
os.makedirs("models", exist_ok=True)

st.set_page_config(page_title="Fish AI - Enterprise Suite", page_icon="🐟", layout="wide")

# ==========================================
# 2. PREMIUM DARK UI (GLASSMORPHISM)
# ==========================================
def apply_pro_styling():
    st.markdown("""
    <style>
    .stApp {{
        background: #0a0c10;
        color: #ffffff;
    }}
    .main-card {{
        background: rgba(255, 255, 255, 0.03);
        backdrop-filter: blur(20px);
        border-radius: 25px; border: 1px solid rgba(0, 194, 255, 0.2);
        padding: 40px; margin-bottom: 25px;
    }}
    .res-box {{
        background: rgba(0, 194, 255, 0.07);
        border-left: 5px solid #00C2FF;
        border-radius: 10px; padding: 20px;
    }}
    div.stButton > button {{
        background: linear-gradient(90deg, #00C2FF, #0072FF);
        color: white; border: none; border-radius: 12px; height: 3.8em; font-weight: bold; width: 100%;
        transition: 0.3s ease;
    }}
    div.stButton > button:hover {{ transform: scale(1.02); box-shadow: 0 0 20px rgba(0,194,255,0.4); }}
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
            with st.spinner("🚀 Downloading Expert Neural Weights from Hugging Face..."):
                r = requests.get(HF_EXPERT_URL, stream=True)
                with open(MODEL_PATH, "wb") as f:
                    for chunk in r.iter_content(chunk_size=8192):
                        f.write(chunk)
        except Exception as e: return None, f"Sync Failed: {str(e)}"
    
    try:
        # ResNet50 Architecture for 21 Classes
        model = models.resnet50(weights=None)
        model.fc = nn.Linear(model.fc.in_features, 21)
        
        # State Dict Cleaning Logic (Industry Standard for SimCLR)
        sd = torch.load(MODEL_PATH, map_location=torch.device('cpu'))
        clean_sd = {k.replace("encoder.", "").replace("model.", ""): v for k, v in sd.items()}
        
        model.load_state_dict(clean_sd, strict=False)
        model.eval()
        return model, "Expert Engine Operational"
    except Exception as e:
        return None, f"Engine Failure: {str(e)}"

expert_model, engine_status = load_expert_engine()

# Verified Class Names
CLASS_NAMES = [
    "Baim", "Bata", "Batasio(tenra)", "Chitul", "Croaker(Poya)", 
    "Hilsha", "Kajoli", "Meni", "Pabda", "Poli", "Puti", 
    "Rita", "Rui", "Rupchada", "Silver Carp", "Telapiya", 
    "carp", "k", "kaikka", "koral", "shrimp"
]

# ==========================================
# 4. AUTH & SESSION
# ==========================================
if 'user' not in st.session_state: st.session_state['user'] = None

with st.sidebar:
    st.markdown("### 🛡️ System Control")
    if st.session_state['user']:
        st.success(f"Access Granted: {st.session_state['user']}")
        app_mode = st.radio("Switch View", ["Neural Analyzer", "Account Settings", "Logout"])
    else:
        app_mode = st.radio("Portal Access", ["Sign In", "Register"])
    
    st.markdown("---")
    st.caption(f"**AI Status:** {engine_status}")
    st.caption("Deployment: v3.1 Enterprise")

# ==========================================
# 5. CORE LOGIC
# ==========================================
if app_mode == "Sign In":
    st.markdown('<div class="main-card"><h2>Professional Login</h2></div>', unsafe_allow_html=True)
    u = st.text_input("Username")
    if st.button("Unlock Dashboard"):
        st.session_state['user'] = u
        st.rerun()

elif app_mode == "Logout":
    st.session_state['user'] = None
    st.rerun()

elif app_mode == "Neural Analyzer":
    st.markdown('<div class="main-card"><h1>Deep Neural Fish Identification</h1><p>Powered by SimCLR Expert Weights</p></div>', unsafe_allow_html=True)
    
    source = st.file_uploader("Upload High-Res Sample", type=["jpg", "png", "jpeg"])
    
    if source:
        col_img, col_data = st.columns([1, 1.2])
        with col_img:
            image = Image.open(source).convert('RGB')
            st.image(image, caption="Analyzed Specimen", use_container_width=True)
        
        with col_data:
            if st.button("RUN EXPERT DIAGNOSTICS"):
                if expert_model:
                    with st.spinner("Mapping Morphology..."):
                        # Industry Normalization
                        transform = transforms.Compose([
                            transforms.Resize((224, 224)),
                            transforms.ToTensor(),
                            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                        ])
                        tensor = transform(image).unsqueeze(0)
                        
                        with torch.no_grad():
                            logits = expert_model(tensor)
                            probs = torch.nn.functional.softmax(logits[0], dim=0)
                            conf, idx = torch.max(probs, 0)
                        
                        # --- THE PRECISION FILTER (রেজাল্ট নির্ভুল করার চাবিকাঠি) ---
                        if conf.item() < 0.75:
                            st.warning("⚠️ **Reliability Alert:** Model confidence is below 75%. The specimen might be poorly lit or a non-indexed species.")
                        
                        st.markdown(f'''
                            <div class="res-box">
                                <h2 style="color: #00C2FF; margin:0;">Identified: {CLASS_NAMES[idx.item()]}</h2>
                                <h3 style="margin:0; font-weight:400;">Confidence: {conf.item()*100:.2f}%</h3>
                            </div>
                        ''', unsafe_allow_html=True)
                        
                        # Data Insights
                        top_vals, top_idxs = torch.topk(probs, 5)
                        analytics = pd.DataFrame({
                            'Fish Species': [CLASS_NAMES[i] for i in top_idxs],
                            'Probability (%)': top_vals.numpy() * 100
                        })
                        st.write("#### Neural Prediction Distribution")
                        st.bar_chart(analytics, x='Fish Species', y='Probability (%)', horizontal=True)

st.markdown('<p style="text-align:center; color:#4a4a4a; margin-top:80px;">© 2026 RIAD AI INDUSTRIES • SECURE CLOUD DEPLOYMENT</p>', unsafe_allow_html=True)
