import streamlit as st
import os
import requests
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import pandas as pd
import webbrowser

# ==========================================
# ১. রিসোর্স এবং ডার্ক থিম
# ==========================================
HF_EXPERT_URL = "https://huggingface.co/riad300/fish-simclr-encoder/resolve/main/fish_expert_weights.pt"
MODEL_PATH = "models/fish_expert_weights.pt"
os.makedirs("models", exist_ok=True)

st.set_page_config(page_title="Fish AI - Google Sync", page_icon="🐟", layout="wide")

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
        backdrop-filter: blur(30px);
        border-radius: 20px; border: 1px solid rgba(66, 133, 244, 0.5); /* Google Blue Border */
        padding: 40px; color: white;
    }
    div.stButton > button {
        background: linear-gradient(90deg, #4285F4, #34A853); /* Google Colors */
        color: white; border-radius: 10px; font-weight: bold; border: none;
    }
    </style>
    """, unsafe_allow_html=True)

apply_google_theme()

# ==========================================
# ২. সঠিক ফোল্ডার সিরিয়াল (Sync with Notebook)
# ==========================================
CLASS_NAMES = sorted([
    "Baim", "Bata", "Batasio(tenra)", "Chitul", "Croaker(Poya)", 
    "Hilsha", "Kajoli", "Meni", "Pabda", "Poli", "Puti", 
    "Rita", "Rui", "Rupchada", "Silver Carp", "Telapiya", 
    "carp", "k", "kaikka", "koral", "shrimp"
])

# ==========================================
# ৩. আপনার নোটবুকের সিমলার মডেল আর্কিটেকচার
# ==========================================
class SimCLR(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = models.resnet50(weights=None)
        self.encoder.fc = nn.Identity() 

@st.cache_resource
def load_engine():
    if not os.path.exists(MODEL_PATH):
        r = requests.get(HF_EXPERT_URL, stream=True)
        with open(MODEL_PATH, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192): f.write(chunk)
    
    try:
        base = SimCLR()
        model = nn.Sequential(base.encoder, nn.Linear(2048, 21))
        sd = torch.load(MODEL_PATH, map_location=torch.device('cpu'))
        # Key cleaning for Seq model
        new_sd = {k.replace("0.encoder.", "0."): v for k, v in sd.items()}
        model.load_state_dict(new_sd, strict=False)
        model.eval()
        return model
    except: return None

# ==========================================
# ৪. অ্যাপ লজিক ও লগইন
# ==========================================
if 'logged_in' not in st.session_state: st.session_state['logged_in'] = False

if not st.session_state['logged_in']:
    st.markdown('<div class="main-card"><h2>🔐 Secure Expert Login</h2></div>', unsafe_allow_html=True)
    user_id = st.text_input("Admin ID")
    if st.button("Access Dashboard"):
        if user_id:
            st.session_state['logged_in'] = True
            st.rerun()
else:
    with st.sidebar:
        st.success("Authorized: Admin")
        if st.button("Logout"):
            st.session_state['logged_in'] = False
            st.rerun()

    st.markdown('<div class="main-card"><h1>🐟 Fish AI <span style="color:#4285F4">Google</span> Engine</h1><p>Combined Neural & Web Intelligence</p></div>', unsafe_allow_html=True)
    
    file = st.file_uploader("Upload Fish Specimen", type=["jpg", "png", "jpeg"])
    
    if file:
        img = Image.open(file).convert('RGB')
        col1, col2 = st.columns([1, 1.2])
        
        with col1:
            st.image(img, use_container_width=True, caption="Target Specimen")
        
        with col2:
            if st.button("🚀 ANALYZE WITH GOOGLE PRECISION"):
                model = load_engine()
                if model:
                    with st.spinner("Decoding Morphology..."):
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
                        
                        prediction = CLASS_NAMES[idx.item()]
                        
                        # রেজাল্ট ডিসপ্লে
                        st.markdown(f"""
                            <div style="background:rgba(66,133,244,0.1); border:1px solid #4285F4; padding:25px; border-radius:15px;">
                                <h2 style="color:#4285F4; margin:0;">Specimen Name: {prediction}</h2>
                                <h3 style="margin:0;">Precision: {conf.item()*100:.2f}%</h3>
                            </div>
                        """, unsafe_allow_html=True)

                        # সরাসরি গুগল সার্চের ভেরিফিকেশন লিঙ্ক
                        st.write("---")
                        st.subheader("🔍 Google Intelligence Verification")
                        google_search_url = f"https://www.google.com/search?q={prediction}+fish+bangladesh"
                        st.info(f"এই মাছটি সম্পর্কে নিশ্চিত হতে সরাসরি গুগল ডেটাবেজে দেখুন।")
                        st.markdown(f'''<a href="{google_search_url}" target="_blank" style="text-decoration:none;">
                            <button style="background-color:#EA4335; color:white; padding:10px 20px; border:none; border-radius:5px; cursor:pointer;">
                                Verify Results on Google Search
                            </button>
                        </a>''', unsafe_allow_html=True)
                        
                        # টপ ৫ ডিস্ট্রিবিউশন
                        top5_p, top5_i = torch.topk(prob, 5)
                        df = pd.DataFrame({'Species': [CLASS_NAMES[i] for i in top5_i], 'Match %': top5_p.numpy()*100})
                        st.bar_chart(df, x='Species', y='Match %')

st.markdown('<p style="text-align:center; color:gray; margin-top:80px;">© 2026 RIAD AI INDUSTRIES • CLOUD PRECISION BUILD</p>', unsafe_allow_html=True)
