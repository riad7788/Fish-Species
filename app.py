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
# 1. SMART MODEL SELECTION (Hugging Face)
# ==========================================
# আমরা এখন 'fish_expert_weights.pt' ব্যবহার করছি ভালো রেজাল্টের জন্য
HF_EXPERT_URL = "https://huggingface.co/riad300/fish-simclr-encoder/resolve/main/fish_expert_weights.pt"
MODEL_LOCAL_PATH = "models/fish_expert_weights.pt"
os.makedirs("models", exist_ok=True)

st.set_page_config(page_title="Fish AI Expert", page_icon="🐟", layout="wide")

# ==========================================
# 2. UI THEME (FIXED & PROFESSIONAL)
# ==========================================
def apply_theme():
    st.markdown("""
    <style>
    .stApp {
        background: linear-gradient(rgba(0,0,0,0.85), rgba(0,0,0,0.85)), 
                    url("https://images.unsplash.com/photo-1524704654690-b56c05c78a00?q=80&w=2069");
        background-size: cover; background-attachment: fixed;
    }
    .glass-card {
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(15px);
        border-radius: 20px; border: 1px solid rgba(255, 255, 255, 0.1);
        padding: 30px; margin-bottom: 20px; color: white; text-align: center;
    }
    div.stButton > button {
        background: linear-gradient(90deg, #00C2FF, #0072FF);
        color: white; border-radius: 12px; font-weight: bold; width: 100%; height: 3.5em; border: none;
    }
    </style>
    """, unsafe_allow_html=True)

apply_theme()

# ==========================================
# 3. EXPERT ENGINE (LOADS WEIGHTS CORRECTLY)
# ==========================================
@st.cache_resource
def load_expert_model():
    if not os.path.exists(MODEL_LOCAL_PATH):
        try:
            r = requests.get(HF_EXPERT_URL)
            with open(MODEL_LOCAL_PATH, "wb") as f:
                f.write(r.content)
        except: return None, "Hugging Face Sync Failed"
    
    try:
        # ইন্ডাস্ট্রি স্ট্যান্ডার্ড ResNet50
        model = models.resnet50(weights=None)
        model.fc = nn.Linear(model.fc.in_features, 21) # আপনার ২১টি ক্লাস
        
        state_dict = torch.load(MODEL_LOCAL_PATH, map_location=torch.device('cpu'))
        
        # কী-মিসম্যাচ ফিক্সিং (Expert Weight loading)
        new_state_dict = {}
        for k, v in state_dict.items():
            name = k.replace("encoder.", "").replace("model.", "") # সব রকম প্রিফিক্স ক্লিন করা হচ্ছে
            new_state_dict[name] = v
            
        model.load_state_dict(new_state_dict, strict=False)
        model.eval()
        return model, "Expert Weights Loaded"
    except Exception as e:
        return None, f"Loading Error: {str(e)}"

model, model_info = load_expert_model()

# সঠিক ক্লাসের অর্ডার
CLASS_NAMES = [
    "Baim", "Bata", "Batasio(tenra)", "Chitul", "Croaker(Poya)", 
    "Hilsha", "Kajoli", "Meni", "Pabda", "Poli", "Puti", 
    "Rita", "Rui", "Rupchada", "Silver Carp", "Telapiya", 
    "carp", "k", "kaikka", "koral", "shrimp"
]

# ==========================================
# 4. DASHBOARD & AUTH
# ==========================================
if 'user' not in st.session_state: st.session_state['user'] = None

with st.sidebar:
    st.title("🐟 Fish AI Pro")
    if st.session_state['user']:
        st.success(f"Expert: {st.session_state['user']}")
        nav = st.radio("System", ["Dashboard", "Logout"])
    else:
        nav = st.radio("System", ["Login"])
    st.markdown("---")
    st.write(f"**Model Type:** {model_info}")

# --- PAGES ---
if nav == "Login":
    st.markdown('<div class="glass-card"><h2>Expert Login</h2></div>', unsafe_allow_html=True)
    u = st.text_input("Username")
    if st.button("Access Engine"):
        st.session_state['user'] = u
        st.rerun()

elif nav == "Logout":
    st.session_state['user'] = None
    st.rerun()

elif nav == "Dashboard":
    st.markdown('<div class="glass-card"><h1>Deep Neural Fish Analysis</h1></div>', unsafe_allow_html=True)
    
    file = st.file_uploader("Upload Specimen Image", type=["jpg", "png", "jpeg"])
    if file:
        col1, col2 = st.columns([1, 1.2])
        with col1:
            img = Image.open(file).convert('RGB')
            st.image(img, caption="Input Specimen", use_container_width=True)
        
        with col2:
            if st.button("🚀 EXECUTE EXPERT ANALYSIS"):
                if model:
                    with st.spinner("Processing Neural Layers..."):
                        # High Precision Transform
                        transform = transforms.Compose([
                            transforms.Resize((224, 224)),
                            transforms.ToTensor(),
                            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                        ])
                        input_tensor = transform(img).unsqueeze(0)
                        
                        with torch.no_grad():
                            output = model(input_tensor)
                            prob = torch.nn.functional.softmax(output[0], dim=0)
                            conf, idx = torch.max(prob, 0)
                        
                        # রেজাল্ট কার্ড
                        st.markdown(f'''
                            <div class="glass-card" style="border: 2px solid #00C2FF;">
                                <h2 style="color: #00C2FF; margin-bottom: 0px;">Species: {CLASS_NAMES[idx.item()]}</h2>
                                <h3>Confidence Match: {conf.item()*100:.2f}%</h3>
                            </div>
                        ''', unsafe_allow_html=True)
                        
                        # প্রব্যাবিলিটি গ্রাফ (ভুল কমানোর জন্য)
                        top5_p, top5_i = torch.topk(prob, 5)
                        df = pd.DataFrame({
                            'Species': [CLASS_NAMES[i] for i in top5_i],
                            'Probability (%)': top5_p.numpy() * 100
                        })
                        st.write("#### Confidence Distribution (Top 5)")
                        st.bar_chart(df, x='Species', y='Probability (%)', horizontal=True)

st.markdown('<p style="text-align:center; color:gray; margin-top:100px;">© 2026 Fish Expert Systems | Market Build</p>', unsafe_allow_html=True)
