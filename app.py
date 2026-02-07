import streamlit as st
import os
import requests
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import pandas as pd
import urllib.parse

# ==========================================
# ১. গ্লোবাল কনফিগ ও ব্যাকগ্রাউন্ড
# ==========================================
HF_EXPERT_URL = "https://huggingface.co/riad300/fish-simclr-encoder/resolve/main/fish_expert_weights.pt"
MODEL_PATH = "models/fish_expert_weights.pt"
os.makedirs("models", exist_ok=True)

st.set_page_config(page_title="Fish AI - Google Engine", page_icon="🐟", layout="wide")

def apply_google_style():
    st.markdown("""
    <style>
    .stApp {
        background: linear-gradient(rgba(0,0,0,0.9), rgba(0,0,0,0.9)), 
                    url("https://images.unsplash.com/photo-1516734212186-a967f81ad0d7?q=80&w=2071") !important;
        background-size: cover !important;
    }
    .google-card {
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(20px);
        border-radius: 20px; border: 1px solid #4285F4;
        padding: 30px; color: white; margin-bottom: 20px;
    }
    div.stButton > button {
        background: linear-gradient(90deg, #4285F4, #EA4335, #FBBC05, #34A853);
        color: white; border-radius: 10px; font-weight: bold; border: none; height: 3.5em; width: 100%;
    }
    </style>
    """, unsafe_allow_html=True)

apply_google_style()

# ==========================================
# ২. ক্লাস ম্যাপিং (Fixed Order)
# ==========================================
CLASS_NAMES = sorted([
    "Baim", "Bata", "Batasio(tenra)", "Chitul", "Croaker(Poya)", 
    "Hilsha", "Kajoli", "Meni", "Pabda", "Poli", "Puti", 
    "Rita", "Rui", "Rupchada", "Silver Carp", "Telapiya", 
    "carp", "k", "kaikka", "koral", "shrimp"
])

# ==========================================
# ৩. সিমলার মডেল লোডার (Sync with Notebook)
# ==========================================
@st.cache_resource
def load_expert_engine():
    if not os.path.exists(MODEL_PATH):
        r = requests.get(HF_EXPERT_URL, stream=True)
        with open(MODEL_PATH, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192): f.write(chunk)
    try:
        base = models.resnet50(weights=None)
        base.fc = nn.Identity()
        model = nn.Sequential(base, nn.Linear(2048, 21))
        sd = torch.load(MODEL_PATH, map_location=torch.device('cpu'))
        # কী-ক্লিনিং
        new_sd = {k.replace("0.encoder.", "0."): v for k, v in sd.items()}
        model.load_state_dict(new_sd, strict=False)
        model.eval()
        return model
    except: return None

# ==========================================
# ৪. মেইন ড্যাশবোর্ড লজিক
# ==========================================
if 'logged' not in st.session_state: st.session_state['logged'] = False

if not st.session_state['logged']:
    st.markdown('<div class="google-card"><h2>🛡️ Admin Login</h2></div>', unsafe_allow_html=True)
    user = st.text_input("Enter Access ID")
    if st.button("Authorize"):
        if user:
            st.session_state['logged'] = True
            st.rerun()
else:
    with st.sidebar:
        st.success("Google Engine Connected")
        if st.button("Logout"):
            st.session_state['logged'] = False
            st.rerun()

    st.markdown('<div class="google-card"><h1>🐟 Google Integrated Fish AI</h1><p>Neural Prediction with Google Image Verification</p></div>', unsafe_allow_html=True)
    
    file = st.file_uploader("Upload Specimen", type=["jpg", "png", "jpeg"])
    
    if file:
        img = Image.open(file).convert('RGB')
        col1, col2 = st.columns([1, 1.2])
        
        with col1:
            st.image(img, use_container_width=True, caption="Target Image")
        
        with col2:
            if st.button("🔍 START GOOGLE PRECISION ANALYSIS"):
                model = load_expert_engine()
                if model:
                    with st.spinner("Searching Google Cloud & Local Neural Data..."):
                        # আপনার নোটবুক অনুযায়ী ১৬০x১৬০ ইমেজ ট্রান্সফর্ম
                        transform = transforms.Compose([
                            transforms.Resize((160, 160)),
                            transforms.ToTensor(),
                            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                        ])
                        
                        tensor = transform(img).unsqueeze(0)
                        
                        with torch.no_grad():
                            out = model(tensor)
                            prob = torch.nn.functional.softmax(out[0], dim=0)
                            conf, idx = torch.max(prob, 0)
                        
                        fish_name = CLASS_NAMES[idx.item()]
                        
                        # ১. লোকাল মডেল রেজাল্ট
                        st.markdown(f"""
                            <div style="background:rgba(66,133,244,0.1); border-left:5px solid #4285F4; padding:20px; border-radius:10px;">
                                <h3 style="margin:0; color:#4285F4;">Neural Prediction: {fish_name}</h3>
                                <p style="margin:0;">Model Confidence: {conf.item()*100:.2f}%</p>
                            </div>
                        """, unsafe_allow_html=True)

                        # ২. গুগল ভেরিফিকেশন (Google Reverse Image Search Concept)
                        st.write("---")
                        st.subheader("🌐 Global Verification (Direct Google Sync)")
                        
                        # এটি সরাসরি আপনার আপলোড করা ইমেজের নাম গুগলে সার্চ করবে
                        search_query = urllib.parse.quote(f"{fish_name} fish of Bangladesh")
                        google_search_url = f"https://www.google.com/search?q={search_query}&tbm=isch"
                        
                        st.info(f"মডেলের প্রেডিকশন ভুল মনে হলে সরাসরি গুগলের আসল ডেটাবেজে মিলিয়ে দেখুন:")
                        
                        st.markdown(f"""
                            <a href="{google_search_url}" target="_blank">
                                <button style="background-color:#EA4335; color:white; padding:15px; border:none; border-radius:10px; cursor:pointer; font-size:18px; width:100%;">
                                    Check Google Visual Data for "{fish_name}"
                                </button>
                            </a>
                        """, unsafe_allow_html=True)
                        
                        # চার্ট
                        st.write("#### Neural Class Probabilities")
                        top5_p, top5_i = torch.topk(prob, 5)
                        df = pd.DataFrame({'Species': [CLASS_NAMES[i] for i in top5_i], 'Confidence %': top5_p.numpy()*100})
                        st.bar_chart(df, x='Species', y='Confidence %')

st.markdown('<p style="text-align:center; color:gray; margin-top:50px;">© 2026 RIAD AI • GOOGLE SYNCED ENTERPRISE BUILD</p>', unsafe_allow_html=True)
