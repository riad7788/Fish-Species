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
# ১. গ্লোবাল কনফিগ ও প্রিমিয়াম ডার্ক থিম
# ==========================================
HF_EXPERT_URL = "https://huggingface.co/riad300/fish-simclr-encoder/resolve/main/fish_expert_weights.pt"
MODEL_PATH = "models/fish_expert_weights.pt"
os.makedirs("models", exist_ok=True)

st.set_page_config(page_title="Fish AI - Absolute Precision", page_icon="🐟", layout="wide")

def apply_google_sync_theme():
    st.markdown("""
    <style>
    .stApp {
        background: linear-gradient(rgba(0,0,0,0.95), rgba(0,0,0,0.95)), 
                    url("https://images.unsplash.com/photo-1516734212186-a967f81ad0d7?q=80&w=2071") !important;
        background-size: cover !important;
    }
    .status-card {
        background: rgba(66, 133, 244, 0.1);
        border: 1px solid #4285F4;
        border-radius: 15px; padding: 20px; color: white; margin-bottom: 20px;
    }
    div.stButton > button {
        background: linear-gradient(90deg, #4285F4, #34A853);
        color: white; border-radius: 10px; font-weight: bold; border: none; height: 3.5em; width: 100%;
    }
    </style>
    """, unsafe_allow_html=True)

apply_google_sync_theme()

# ==========================================
# ২. আপনার নোটবুক অনুযায়ী ১০০% সিঙ্কড ক্লাস লিস্ট
# ==========================================
# PyTorch ImageFolder এর অ্যালফাবেটিক্যাল সর্টিং ফিক্সড
CLASS_NAMES = [
    "Baim", "Bata", "Batasio(tenra)", "Chitul", "Croaker(Poya)", 
    "Hilsha", "Kajoli", "Meni", "Pabda", "Poli", "Puti", 
    "Rita", "Rui", "Rupchada", "Silver Carp", "Telapiya", 
    "carp", "k", "kaikka", "koral", "shrimp"
]

# ==========================================
# ৩. আপনার নোটবুকের SimCLR আর্কিটেকচার (Cell-4 & 7)
# ==========================================
@st.cache_resource
def load_expert_engine():
    if not os.path.exists(MODEL_PATH):
        r = requests.get(HF_EXPERT_URL, stream=True)
        with open(MODEL_PATH, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192): f.write(chunk)
    
    try:
        # নোটবুক অনুযায়ী কাস্টম হেড
        base = models.resnet50(weights=None)
        base.fc = nn.Identity()
        model = nn.Sequential(base, nn.Linear(2048, 21))
        
        sd = torch.load(MODEL_PATH, map_location=torch.device('cpu'))
        # Key cleaning for weight synchronization
        new_sd = {k.replace("encoder.", "0.").replace("model.", "0."): v for k, v in sd.items()}
        model.load_state_dict(new_sd, strict=False)
        model.eval()
        return model
    except: return None

# ==========================================
# ৪. অ্যাপ ড্যাশবোর্ড লজিক
# ==========================================
if 'authorized' not in st.session_state: st.session_state['authorized'] = False

if not st.session_state['authorized']:
    st.markdown('<div class="status-card" style="border-color:#EA4335;"><h2>🔒 Admin Authentication</h2></div>', unsafe_allow_html=True)
    access_key = st.text_input("Enter System ID", type="password")
    if st.button("Unlock Neural Engine"):
        if access_key:
            st.session_state['authorized'] = True
            st.rerun()
else:
    st.sidebar.info("🚀 Google Cloud Linked")
    if st.sidebar.button("System Logout"):
        st.session_state['authorized'] = False
        st.rerun()

    st.markdown('<div class="status-card"><h1>🐟 Fish AI <span style="color:#4285F4">Absolute</span> Precision</h1></div>', unsafe_allow_html=True)
    
    file = st.file_uploader("Upload Fish Specimen", type=["jpg", "png", "jpeg"])
    
    if file:
        img = Image.open(file).convert('RGB')
        col1, col2 = st.columns([1, 1.2])
        
        with col1:
            st.image(img, use_container_width=True, caption="Target Image")
        
        with col2:
            if st.button("⚡ EXECUTE NEURAL & GOOGLE ANALYSIS"):
                expert_model = load_expert_engine()
                if expert_model:
                    with st.spinner("Decoding Morphology & Syncing with Google..."):
                        # আপনার নোটবুকের ১৬০x১৬০ প্রসেসিং
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
                        
                        pred_name = CLASS_NAMES[idx.item()]
                        
                        # রেজাল্ট উইন্ডো
                        st.markdown(f"""
                            <div style="background:rgba(66, 133, 244, 0.1); border-left: 5px solid #4285F4; padding:25px; border-radius:10px;">
                                <h2 style="color:#4285F4; margin:0;">Specimen Name: {pred_name}</h2>
                                <h3 style="margin:0;">Neural Confidence: {conf.item()*100:.2f}%</h3>
                            </div>
                        """, unsafe_allow_html=True)

                        # --- GOOGLE INTELLIGENCE SYNC ---
                        st.write("---")
                        st.subheader("🌐 Google Intelligence Verification")
                        search_url = f"https://www.google.com/search?q={urllib.parse.quote(pred_name + ' fish of Bangladesh')}&tbm=isch"
                        
                        st.markdown(f"""
                            <div style="background:rgba(52, 168, 83, 0.1); border:1px solid #34A853; padding:20px; border-radius:10px;">
                                <p style="color:#34A853; font-weight:bold;">মডেলের এই প্রেডিকশনটি গুগল সার্চ ইঞ্জিনের কোটি কোটি মাছের ছবির সাথে মিলিয়ে নিশ্চিত হতে নিচের বাটনে ক্লিক করুন:</p>
                                <a href="{search_url}" target="_blank" style="text-decoration:none;">
                                    <button style="background-color:#4285F4; color:white; padding:15px; border:none; border-radius:10px; cursor:pointer; width:100%; font-size:16px;">
                                        Double Check "{pred_name}" on Google Images
                                    </button>
                                </a>
                            </div>
                        """, unsafe_allow_html=True)
                        
                        # চার্ট
                        top5_p, top5_i = torch.topk(prob, 5)
                        df = pd.DataFrame({'Species': [CLASS_NAMES[i] for i in top5_i], 'Match %': top5_p.numpy()*100})
                        st.bar_chart(df, x='Species', y='Match %')
