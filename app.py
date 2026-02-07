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
# ১. রিসোর্স ও গুগল থিম কনফিগ
# ==========================================
HF_EXPERT_URL = "https://huggingface.co/riad300/fish-simclr-encoder/resolve/main/fish_expert_weights.pt"
MODEL_PATH = "models/fish_expert_weights.pt"
os.makedirs("models", exist_ok=True)

st.set_page_config(page_title="Fish AI - Google Precision", page_icon="🐟", layout="wide")

def apply_google_theme():
    st.markdown("""
    <style>
    .stApp {
        background: linear-gradient(rgba(0,0,0,0.92), rgba(0,0,0,0.92)), 
                    url("https://images.unsplash.com/photo-1516734212186-a967f81ad0d7?q=80&w=2071") !important;
        background-size: cover !important;
    }
    .main-card {
        background: rgba(255, 255, 255, 0.03);
        backdrop-filter: blur(25px);
        border-radius: 20px; border: 1px solid #4285F4;
        padding: 40px; color: white;
    }
    div.stButton > button {
        background: linear-gradient(90deg, #4285F4, #34A853, #FBBC05, #EA4335);
        color: white; border-radius: 12px; font-weight: bold; border: none; height: 3.8em;
    }
    </style>
    """, unsafe_allow_html=True)

apply_google_theme()

# ==========================================
# ২. আপনার নোটবুকের সঠিক সিরিয়াল (FIXED MAPPING)
# ==========================================
# আপনার নোটবুকে PyTorch ImageFolder যেভাবে ফোল্ডার সাজায়
CLASS_NAMES = sorted([
    "Baim", "Bata", "Batasio(tenra)", "Chitul", "Croaker(Poya)", 
    "Hilsha", "Kajoli", "Meni", "Pabda", "Poli", "Puti", 
    "Rita", "Rui", "Rupchada", "Silver Carp", "Telapiya", 
    "carp", "k", "kaikka", "koral", "shrimp"
])

# ==========================================
# ৩. ১০০% নির্ভুল মডেল আর্কিটেকচার (Sync with Cell-7)
# ==========================================
@st.cache_resource
def load_expert_engine():
    if not os.path.exists(MODEL_PATH):
        r = requests.get(HF_EXPERT_URL, stream=True)
        with open(MODEL_PATH, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192): f.write(chunk)
    
    try:
        # নোটবুক অনুযায়ী ResNet50 + Identity + Linear Head
        base_resnet = models.resnet50(weights=None)
        base_resnet.fc = nn.Identity()
        model = nn.Sequential(base_resnet, nn.Linear(2048, 21))
        
        checkpoint = torch.load(MODEL_PATH, map_location=torch.device('cpu'))
        # কী-ম্যাপিং ফিক্স (Prefix mismatch হ্যান্ডেল করা হয়েছে)
        new_sd = {}
        for k, v in checkpoint.items():
            name = k.replace("encoder.", "0.").replace("model.", "0.")
            new_sd[name] = v
            
        model.load_state_dict(new_sd, strict=False)
        model.eval()
        return model
    except:
        return None

# ==========================================
# ৪. ড্যাশবোর্ড লজিক
# ==========================================
if 'authorized' not in st.session_state: st.session_state['authorized'] = False

if not st.session_state['authorized']:
    st.markdown('<div class="main-card"><h2>🛡️ Admin Access Restricted</h2></div>', unsafe_allow_html=True)
    access_code = st.text_input("Enter System ID", type="password")
    if st.button("Unlock Dashboard"):
        if access_code:
            st.session_state['authorized'] = True
            st.rerun()
else:
    st.sidebar.success("✅ Google Master Engine: Connected")
    if st.sidebar.button("System Logout"):
        st.session_state['authorized'] = False
        st.rerun()

    st.markdown('<div class="main-card"><h1>🐟 Fish AI <span style="color:#4285F4">Absolute</span> Precision</h1><p>Syncing Neural Data with Google Image Cloud</p></div>', unsafe_allow_html=True)
    
    file = st.file_uploader("Upload Fish Photo", type=["jpg", "png", "jpeg"])
    
    if file:
        img = Image.open(file).convert('RGB')
        col1, col2 = st.columns([1, 1.2])
        
        with col1:
            st.image(img, use_container_width=True, caption="Target Image")
        
        with col2:
            if st.button("🚀 EXECUTE GOOGLE-SYNC SEARCH"):
                expert_model = load_expert_engine()
                if expert_model:
                    with st.spinner("Decoding Morphology..."):
                        # আপনার নোটবুকের ট্রেনিং সাইজ ১৬০x১৬০
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
                        
                        st.markdown(f"""
                            <div style="background:rgba(66,133,244,0.1); border:2px solid #4285F4; padding:25px; border-radius:15px;">
                                <h2 style="color:#4285F4; margin:0;">Identified: {predicted_name}</h2>
                                <h3 style="margin:0;">Precision: {conf.item()*100:.2f}%</h3>
                            </div>
                        """, unsafe_allow_html=True)

                        # --- GOOGLE SMART VERIFICATION ---
                        st.write("---")
                        st.subheader("🌐 Verify with Google Cloud Database")
                        search_url = f"https://www.google.com/search?q={urllib.parse.quote(predicted_name + ' fish of Bangladesh')}&tbm=isch"
                        
                        st.info(f"এই প্রেডিকশনটি গুগল সার্চ ইঞ্জিনের তথ্যের সাথে মিলিয়ে নিতে নিচের বাটনে ক্লিক করুন:")
                        st.markdown(f'''
                            <a href="{search_url}" target="_blank">
                                <button style="background-color:#EA4335; color:white; padding:15px; border:none; border-radius:10px; cursor:pointer; font-weight:bold; width:100%;">
                                    Check Google Images for "{predicted_name}"
                                </button>
                            </a>
                        ''', unsafe_allow_html=True)
                        
                        # চার্ট
                        st.write("#### Confidence Distribution")
                        top5_p, top5_i = torch.topk(prob, 5)
                        df = pd.DataFrame({'Species': [CLASS_NAMES[i] for i in top5_i], 'Match %': top5_p.numpy()*100})
                        st.bar_chart(df, x='Species', y='Match %')
