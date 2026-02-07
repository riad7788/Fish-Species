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
# ১. গ্লোবাল কনফিগ ও প্রো থিম
# ==========================================
HF_EXPERT_URL = "https://huggingface.co/riad300/fish-simclr-encoder/resolve/main/fish_expert_weights.pt"
MODEL_PATH = "models/fish_expert_weights.pt"
os.makedirs("models", exist_ok=True)

st.set_page_config(page_title="Fish AI - Google Engine", page_icon="🐟", layout="wide")

def apply_google_theme():
    st.markdown("""
    <style>
    .stApp {
        background: linear-gradient(rgba(0,0,0,0.92), rgba(0,0,0,0.92)), 
                    url("https://images.unsplash.com/photo-1516734212186-a967f81ad0d7?q=80&w=2071") !important;
        background-size: cover !important;
    }
    .pro-card {
        background: rgba(255, 255, 255, 0.03);
        backdrop-filter: blur(25px);
        border-radius: 20px; border: 1px solid #4285F4;
        padding: 40px; color: white;
    }
    div.stButton > button {
        background: linear-gradient(90deg, #4285F4, #34A853);
        color: white; border-radius: 12px; font-weight: bold; border: none; height: 3.8em;
    }
    </style>
    """, unsafe_allow_html=True)

apply_google_theme()

# ==========================================
# ২. আপনার নোটবুকের সঠিক সর্টেড ম্যাপিং
# ==========================================
CLASS_NAMES = sorted([
    "Baim", "Bata", "Batasio(tenra)", "Chitul", "Croaker(Poya)", 
    "Hilsha", "Kajoli", "Meni", "Pabda", "Poli", "Puti", 
    "Rita", "Rui", "Rupchada", "Silver Carp", "Telapiya", 
    "carp", "k", "kaikka", "koral", "shrimp"
])

# ==========================================
# ৩. মডেল আর্কিটেকচার সিঙ্ক (Sync with Notebook)
# ==========================================
@st.cache_resource
def load_expert_engine():
    if not os.path.exists(MODEL_PATH):
        r = requests.get(HF_EXPERT_URL, stream=True)
        with open(MODEL_PATH, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192): f.write(chunk)
    
    try:
        # নোটবুকের Cell-7 সিকোয়েন্স
        base = models.resnet50(weights=None)
        base.fc = nn.Identity()
        model = nn.Sequential(base, nn.Linear(2048, 21))
        
        sd = torch.load(MODEL_PATH, map_location=torch.device('cpu'))
        # Key cleaning logic
        new_sd = {k.replace("encoder.", "0.").replace("model.", "0."): v for k, v in sd.items()}
        model.load_state_dict(new_sd, strict=False)
        model.eval()
        return model
    except: return None

# ==========================================
# ৪. মেইন অ্যাপ্লিকেশন লজিক
# ==========================================
if 'logged_in' not in st.session_state: st.session_state['logged_in'] = False

if not st.session_state['logged_in']:
    st.markdown('<div class="pro-card"><h2>🔐 Expert Portal Login</h2></div>', unsafe_allow_html=True)
    user_id = st.text_input("Enter Admin ID")
    if st.button("Unlock Dashboard"):
        if user_id:
            st.session_state['logged_in'] = True
            st.rerun()
else:
    st.sidebar.success("✅ Google Master Engine Active")
    if st.sidebar.button("Logout"):
        st.session_state['logged_in'] = False
        st.rerun()

    st.markdown('<div class="pro-card"><h1>🐟 Fish AI <span style="color:#4285F4">Master</span> Search</h1></div>', unsafe_allow_html=True)
    
    file = st.file_uploader("Upload Specimen", type=["jpg", "png", "jpeg"])
    
    if file:
        img = Image.open(file).convert('RGB')
        col1, col2 = st.columns([1, 1.2])
        
        with col1:
            st.image(img, use_container_width=True, caption="Analyzed Specimen")
        
        with col2:
            if st.button("🚀 EXECUTE GOOGLE-SYNC ANALYSIS"):
                engine = load_expert_engine()
                if engine:
                    with st.spinner("Decoding Morphology & Syncing with Google..."):
                        # নোটবুক সাইজ ১৬০x১৬০
                        transform = transforms.Compose([
                            transforms.Resize((160, 160)),
                            transforms.ToTensor(),
                            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                        ])
                        tensor = transform(img).unsqueeze(0)
                        
                        with torch.no_grad():
                            out = engine(tensor)
                            prob = torch.nn.functional.softmax(out[0], dim=0)
                            conf, idx = torch.max(prob, 0)
                        
                        predicted_fish = CLASS_NAMES[idx.item()]
                        
                        st.markdown(f"""
                            <div style="background:rgba(66,133,244,0.1); border:2px solid #4285F4; padding:25px; border-radius:15px;">
                                <h2 style="color:#4285F4; margin:0;">Identity: {predicted_fish}</h2>
                                <h3 style="margin:0;">Precision: {conf.item()*100:.2f}%</h3>
                            </div>
                        """, unsafe_allow_html=True)

                        # --- GOOGLE SMART VERIFICATION ---
                        st.write("---")
                        st.subheader("🌐 Verify with Google Image Database")
                        search_url = f"https://www.google.com/search?q={urllib.parse.quote(predicted_fish + ' fish of Bangladesh')}&tbm=isch"
                        
                        st.info(f"মডেলের প্রেডিকশন শতভাগ নিশ্চিত করতে নিচের বাটনে ক্লিক করে গুগলের ছবির সাথে মিলিয়ে নিন:")
                        st.markdown(f'''
                            <a href="{search_url}" target="_blank">
                                <button style="background-color:#EA4335; color:white; padding:15px; border:none; border-radius:10px; cursor:pointer; font-weight:bold; width:100%; border: 2px solid white;">
                                    🔍 Double Check "{predicted_fish}" on Google Lens
                                </button>
                            </a>
                        ''', unsafe_allow_html=True)
