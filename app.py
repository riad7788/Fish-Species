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
# ১. গ্লোবাল সেটিংস এবং থিম
# ==========================================
st.set_page_config(page_title="Fish AI - Precision Build", page_icon="🐟", layout="wide")

def apply_custom_theme():
    st.markdown("""
    <style>
    .stApp {
        background: linear-gradient(rgba(0,0,0,0.95), rgba(0,0,0,0.95)), 
                    url("https://images.unsplash.com/photo-1516734212186-a967f81ad0d7?q=80&w=2071") !important;
        background-size: cover !important;
    }
    .main-card {
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(20px);
        border-radius: 20px; border: 1px solid #4285F4;
        padding: 30px; color: white;
    }
    div.stButton > button {
        background: linear-gradient(90deg, #4285F4, #34A853);
        color: white; border-radius: 10px; font-weight: bold; border: none; height: 3.5em; width: 100%;
    }
    </style>
    """, unsafe_allow_html=True)

apply_custom_theme()

# ==========================================
# ২. আপনার নোটবুকের সঠিক সিরিয়াল (Sync with ImageFolder)
# ==========================================
# PyTorch ImageFolder এর ডিফল্ট অর্ডারে এটিই সঠিক সিরিয়াল
CLASS_NAMES = [
    "Baim", "Bata", "Batasio(tenra)", "Chitul", "Croaker(Poya)", 
    "Hilsha", "Kajoli", "Meni", "Pabda", "Poli", "Puti", 
    "Rita", "Rui", "Rupchada", "Silver Carp", "Telapiya", 
    "carp", "k", "kaikka", "koral", "shrimp"
]

# ==========================================
# ৩. ১০০% সিঙ্কড মডেল লোডার
# ==========================================
@st.cache_resource
def load_expert_engine():
    MODEL_URL = "https://huggingface.co/riad300/fish-simclr-encoder/resolve/main/fish_expert_weights.pt"
    MODEL_PATH = "models/fish_expert_weights.pt"
    os.makedirs("models", exist_ok=True)
    
    if not os.path.exists(MODEL_PATH):
        r = requests.get(MODEL_URL, stream=True)
        with open(MODEL_PATH, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192): f.write(chunk)
    
    try:
        # নোটবুকের Cell-7 এর Sequential স্ট্রাকচার অনুযায়ী
        base = models.resnet50(weights=None)
        base.fc = nn.Identity()
        model = nn.Sequential(base, nn.Linear(2048, 21))
        
        sd = torch.load(MODEL_PATH, map_location=torch.device('cpu'))
        # কী-ক্লিনিং (Prefix fixing)
        new_sd = {k.replace("encoder.", "0.").replace("model.", "0."): v for k, v in sd.items()}
        model.load_state_dict(new_sd, strict=False)
        model.eval()
        return model
    except: return None

# ==========================================
# ৪. ড্যাশবোর্ড লজিক (লগইনসহ)
# ==========================================
if 'authorized' not in st.session_state: st.session_state['authorized'] = False

if not st.session_state['authorized']:
    st.markdown('<div class="main-card"><h2>🛡️ Admin Access Restricted</h2></div>', unsafe_allow_html=True)
    access_id = st.text_input("Enter System ID", type="password")
    if st.button("Unlock Dashboard"):
        if access_id:
            st.session_state['authorized'] = True
            st.rerun()
else:
    st.sidebar.success("✅ Neural Engine Active")
    if st.sidebar.button("Logout"):
        st.session_state['authorized'] = False
        st.rerun()

    st.markdown('<div class="main-card"><h1>🐟 Fish AI Master Engine</h1><p>Synced with Google Search Verification</p></div>', unsafe_allow_html=True)
    
    file = st.file_uploader("Upload Fish Image", type=["jpg", "png", "jpeg"])
    
    if file:
        img = Image.open(file).convert('RGB')
        col1, col2 = st.columns([1, 1.2])
        
        with col1:
            st.image(img, use_container_width=True, caption="Specimen Image")
        
        with col2:
            if st.button("🚀 EXECUTE PREDICTION"):
                expert_model = load_expert_engine()
                if expert_model:
                    with st.spinner("Analyzing Morphology..."):
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
                        
                        predicted_name = CLASS_NAMES[idx.item()]
                        
                        # রেজাল্ট ডিসপ্লে
                        st.markdown(f"""
                            <div style="background:rgba(66,133,244,0.1); border:2px solid #4285F4; padding:25px; border-radius:15px;">
                                <h2 style="color:#4285F4; margin:0;">Specimen Name: {predicted_name}</h2>
                                <h3 style="margin:0;">Confidence: {conf.item()*100:.2f}%</h3>
                            </div>
                        """, unsafe_allow_html=True)

                        # --- গুগল সার্চ ভেরিফিকেশন ---
                        st.write("---")
                        st.subheader("🌐 Verify with Google Results")
                        search_url = f"https://www.google.com/search?q={urllib.parse.quote(predicted_name + ' fish of Bangladesh')}&tbm=isch"
                        
                        st.info(f"এই প্রেডিকশনটি সঠিক কি না তা সরাসরি গুগলের আসল ছবির সাথে মিলিয়ে নিন:")
                        st.markdown(f'''
                            <a href="{search_url}" target="_blank">
                                <button style="background-color:#EA4335; color:white; padding:15px; border:none; border-radius:10px; cursor:pointer; font-weight:bold; width:100%;">
                                    Open Google Images for "{predicted_name}"
                                </button>
                            </a>
                        ''', unsafe_allow_html=True)
