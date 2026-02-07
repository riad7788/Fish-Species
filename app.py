import streamlit as st
import os
import requests
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import pandas as pd

# ==========================================
# ১. রিসোর্স এবং ব্যাকগ্রাউন্ড
# ==========================================
HF_EXPERT_URL = "https://huggingface.co/riad300/fish-simclr-encoder/resolve/main/fish_expert_weights.pt"
MODEL_PATH = "models/fish_expert_weights.pt"
os.makedirs("models", exist_ok=True)

st.set_page_config(page_title="Fish AI - Google Enhanced", page_icon="🐟", layout="wide")

def apply_theme():
    st.markdown("""
    <style>
    .stApp {
        background: linear-gradient(rgba(0,0,0,0.9), rgba(0,0,0,0.9)), 
                    url("https://images.unsplash.com/photo-1516734212186-a967f81ad0d7?q=80&w=2071") !important;
        background-size: cover !important;
        background-attachment: fixed !important;
    }
    .main-card {
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(25px);
        border-radius: 20px; border: 1px solid rgba(0, 194, 255, 0.4);
        padding: 30px; color: white;
    }
    </style>
    """, unsafe_allow_html=True)

apply_theme()

# ==========================================
# ২. আপনার নোটবুকের সঠিক সিরিয়াল (Sync with ImageFolder)
# ==========================================
CLASS_NAMES = [
    "Baim", "Bata", "Batasio(tenra)", "Chitul", "Croaker(Poya)", 
    "Hilsha", "Kajoli", "Meni", "Pabda", "Poli", "Puti", 
    "Rita", "Rui", "Rupchada", "Silver Carp", "Telapiya", 
    "carp", "k", "kaikka", "koral", "shrimp"
]

# ==========================================
# ৩. আপনার নোটবুকের SimCLR মডেল লোড
# ==========================================
@st.cache_resource
def load_expert_engine():
    if not os.path.exists(MODEL_PATH):
        r = requests.get(HF_EXPERT_URL, stream=True)
        with open(MODEL_PATH, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192): f.write(chunk)
    
    try:
        # নোটবুকের Cell-7 এর classifier সিকোয়েন্স
        base_resnet = models.resnet50(weights=None)
        base_resnet.fc = nn.Identity() 
        model = nn.Sequential(base_resnet, nn.Linear(2048, 21))
        
        sd = torch.load(MODEL_PATH, map_location=torch.device('cpu'))
        # কী-ক্লিনিং ফিক্স
        new_sd = {k.replace("0.encoder.", "0."): v for k, v in sd.items()}
        model.load_state_dict(new_sd, strict=False)
        model.eval()
        return model
    except: return None

# ==========================================
# ৪. লগইন সিস্টেম (Security)
# ==========================================
if 'auth' not in st.session_state: st.session_state['auth'] = False

if not st.session_state['auth']:
    st.markdown('<div class="main-card"><h2>🛡️ Expert Login Required</h2></div>', unsafe_allow_html=True)
    username = st.text_input("Enter Admin ID")
    if st.button("Unlock Dashboard"):
        if username:
            st.session_state['auth'] = True
            st.rerun()
else:
    with st.sidebar:
        st.success("Authorized: Riad")
        if st.button("Logout"):
            st.session_state['auth'] = False
            st.rerun()

    st.markdown('<div class="main-card"><h1>🐟 Google-Linked Fish Analyzer</h1></div>', unsafe_allow_html=True)
    
    file = st.file_uploader("Upload Fish Specimen", type=["jpg", "png", "jpeg"])
    
    if file:
        img = Image.open(file).convert('RGB')
        col1, col2 = st.columns(2)
        
        with col1:
            st.image(img, use_container_width=True, caption="Specimen for Analysis")
        
        with col2:
            if st.button("🚀 EXECUTE DUAL ANALYSIS"):
                expert_model = load_expert_engine()
                if expert_model:
                    with st.spinner("Decoding Neural Patterns..."):
                        # আপনার নোটবুক অনুযায়ী ১৬০x১৬০ সাইজ
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
                        
                        prediction = CLASS_NAMES[idx.item()]
                        accuracy = conf.item()*100
                        
                        # রেজাল্ট ডিসপ্লে
                        st.markdown(f"""
                            <div style="background:rgba(0,194,255,0.1); border:1px solid #00C2FF; padding:20px; border-radius:15px;">
                                <h2 style="color:#00C2FF; margin:0;">Identified: {prediction}</h2>
                                <h3 style="margin:0;">Neural Confidence: {accuracy:.2f}%</h3>
                            </div>
                        """, unsafe_allow_html=True)
                        
                        # গুগল ভেরিফিকেশন লিঙ্ক
                        st.write("---")
                        st.info(f"🔍 **Google Verification:**")
                        google_url = f"https://www.google.com/search?q={prediction}+fish+of+bangladesh&tbm=isch"
                        st.markdown(f'[Verify "{prediction}" on Google Images]({google_url})')
                        
                        # টপ ৫ ডিস্ট্রিবিউশন
                        top5_p, top5_i = torch.topk(prob, 5)
                        df = pd.DataFrame({'Fish': [CLASS_NAMES[i] for i in top5_i], 'Match %': top5_p.numpy()*100})
                        st.bar_chart(df, x='Fish', y='Match %')

st.markdown('<p style="text-align:center; color:gray; margin-top:80px;">© 2026 RIAD AI • GOOGLE SYNCED ENGINE</p>', unsafe_allow_html=True)
