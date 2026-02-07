import streamlit as st
import os
import requests
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import pandas as pd

# ==========================================
# ১. নোটবুক অনুযায়ী সঠিক আর্কিটেকচার (Cell-7)
# ==========================================
@st.cache_resource
def load_expert_engine():
    MODEL_URL = "https://huggingface.co/riad300/fish-simclr-encoder/resolve/main/fish_expert_weights.pt"
    SAVE_PATH = "models/fish_expert_weights.pt"
    os.makedirs("models", exist_ok=True)

    if not os.path.exists(SAVE_PATH):
        r = requests.get(MODEL_URL, stream=True)
        with open(SAVE_PATH, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192): f.write(chunk)

    try:
        # নোটবুকের কাস্টম ক্লাসিফায়ার স্ট্রাকচার
        base_resnet = models.resnet50(weights=None)
        base_resnet.fc = nn.Identity() 
        model = nn.Sequential(base_resnet, nn.Linear(2048, 21))
        
        checkpoint = torch.load(SAVE_PATH, map_location=torch.device('cpu'))
        
        # কী-ম্যাপিং ফিক্স (নোটবুকের Sequential মডেলের সাথে সিঙ্ক)
        new_sd = {}
        for k, v in checkpoint.items():
            # নোটবুকের 'classifier' ভেরিয়েবল অনুযায়ী নাম পরিবর্তন
            name = k.replace("0.encoder.", "0.") 
            new_sd[name] = v
        
        model.load_state_dict(new_sd, strict=False)
        model.eval()
        return model
    except: return None

# ==========================================
# ২. ডিকশনারি ম্যাপিং (এটিই ভুল হচ্ছিল)
# ==========================================
# আপনার নোটবুকের ডেটাসেট অনুযায়ী পাইথন যেভাবে ইন্ডেক্স করে:
CLASS_NAMES = [
    "Baim", "Bata", "Batasio(tenra)", "Chitul", "Croaker(Poya)", 
    "Hilsha", "Kajoli", "Meni", "Pabda", "Poli", "Puti", 
    "Rita", "Rui", "Rupchada", "Silver Carp", "Telapiya", 
    "carp", "k", "kaikka", "koral", "shrimp"
]

# ==========================================
# ৩. ইউআই ও লগইন সিস্টেম
# ==========================================
st.set_page_config(page_title="Fish AI Expert", page_icon="🐟", layout="wide")

def apply_ui():
    st.markdown("""
    <style>
    .stApp {
        background: linear-gradient(rgba(0,0,0,0.85), rgba(0,0,0,0.85)), 
                    url("https://images.unsplash.com/photo-1516734212186-a967f81ad0d7?q=80&w=2071") !important;
        background-size: cover !important;
        background-attachment: fixed !important;
    }
    .main-card {
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(15px);
        border-radius: 20px; border: 1px solid rgba(0, 194, 255, 0.2);
        padding: 30px; color: white;
    }
    </style>
    """, unsafe_allow_html=True)

apply_ui()

if 'user' not in st.session_state: st.session_state['user'] = None

# লগইন অপশন ফিরিয়ে আনা হয়েছে
if not st.session_state['user']:
    st.markdown('<div class="main-card"><h2>🔒 Expert Portal Login</h2></div>', unsafe_allow_html=True)
    user_id = st.text_input("Username")
    if st.button("Access System"):
        if user_id:
            st.session_state['user'] = user_id
            st.rerun()
else:
    # ড্যাশবোর্ড
    with st.sidebar:
        st.success(f"Verified: {st.session_state['user']}")
        if st.button("Logout"):
            st.session_state['user'] = None
            st.rerun()

    st.markdown('<div class="main-card"><h1>🐟 Deep Neural Fish Analyzer</h1></div>', unsafe_allow_html=True)
    
    file = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])
    
    if file:
        col1, col2 = st.columns(2)
        img = Image.open(file).convert('RGB')
        
        with col1:
            st.image(img, use_container_width=True)
        
        with col2:
            if st.button("🚀 START ANALYSIS"):
                model = load_expert_engine()
                if model:
                    # নোটবুক অনুযায়ী ১৬০x১৬০ ইমেজ ট্রান্সফর্ম (Cell-2)
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
                    
                    st.markdown(f"""
                        <div style="border:2px solid #00C2FF; border-radius:15px; padding:20px; background:rgba(0,194,255,0.1);">
                            <h2 style="color:#00C2FF;">Identified: {CLASS_NAMES[idx.item()]}</h2>
                            <h3>Match: {conf.item()*100:.2f}%</h3>
                        </div>
                    """, unsafe_allow_html=True)
                    
                    # টপ ৫ চার্ট
                    top5_p, top5_i = torch.topk(prob, 5)
                    df = pd.DataFrame({'Fish': [CLASS_NAMES[i] for i in top5_i], 'Confidence (%)': top5_p.numpy()*100})
                    st.bar_chart(df, x='Fish', y='Confidence (%)')
