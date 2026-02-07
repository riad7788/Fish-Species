import streamlit as st
import os
import requests
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import pandas as pd

# ==========================================
# ১. রিসোর্স ও মডেল পাথ কনফিগ
# ==========================================
HF_EXPERT_URL = "https://huggingface.co/riad300/fish-simclr-encoder/resolve/main/fish_expert_weights.pt"
MODEL_PATH = "models/fish_expert_weights.pt"
os.makedirs("models", exist_ok=True)

st.set_page_config(page_title="Fish AI - Expert Build", page_icon="🐟", layout="wide")

# ==========================================
# ২. সঠিক ক্লাস লিস্ট (নোটবুকের ট্রেনিং সিকোয়েন্স অনুযায়ী)
# ==========================================
CLASS_NAMES = [
    "Baim", "Bata", "Batasio(tenra)", "Chitul", "Croaker(Poya)", 
    "Hilsha", "Kajoli", "Meni", "Pabda", "Poli", "Puti", 
    "Rita", "Rui", "Rupchada", "Silver Carp", "Telapiya", 
    "carp", "k", "kaikka", "koral", "shrimp"
]

# ==========================================
# ৩. ইউআই ডিজাইন ও ব্যাকগ্রাউন্ড
# ==========================================
def apply_ui():
    st.markdown("""
    <style>
    .stApp {
        background: linear-gradient(rgba(0,0,0,0.85), rgba(0,0,0,0.85)), 
                    url("https://images.unsplash.com/photo-1524704654690-b56c05c78a00?q=80&w=2069") !important;
        background-size: cover !important;
        background-attachment: fixed !important;
    }
    .glass-card {
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(15px);
        border-radius: 20px; border: 1px solid rgba(0, 194, 255, 0.2);
        padding: 30px; margin-bottom: 20px; color: white;
    }
    div.stButton > button {
        background: linear-gradient(90deg, #00C2FF, #0072FF);
        color: white; border-radius: 12px; height: 3.5em; font-weight: bold; width: 100%; border: none;
    }
    </style>
    """, unsafe_allow_html=True)

apply_ui()

# ==========================================
# ৪. নোটবুক সিঙ্কড মডেল লোডার (FIXED)
# ==========================================
@st.cache_resource
def load_expert_engine():
    # মডেল ডাউনলোড
    if not os.path.exists(MODEL_PATH):
        try:
            r = requests.get(HF_EXPERT_URL, stream=True)
            with open(MODEL_PATH, "wb") as f:
                for chunk in r.iter_content(chunk_size=8192): f.write(chunk)
        except Exception as e: return None

    try:
        # নোটবুকের Cell-7 অনুযায়ী আর্কিটেকচার তৈরি
        base_resnet = models.resnet50(weights=None)
        base_resnet.fc = nn.Identity() # নোটবুকের প্রোজেকশন হেডের আগের অবস্থা
        
        # আপনার ট্রেনিং করা ২১টি ক্লাসের ক্লাসিফায়ার
        model = nn.Sequential(
            base_resnet,
            nn.Linear(2048, 21)
        )
        
        # স্টেট ডিকশনারি লোড করা (Prefix cleaning সহ)
        checkpoint = torch.load(MODEL_PATH, map_location=torch.device('cpu'))
        
        # যদি ডিকশনারির কী-গুলোতে 'encoder.' বা অন্য কিছু থাকে তা রিমুভ করা
        new_sd = {}
        for k, v in checkpoint.items():
            name = k.replace("encoder.", "0.").replace("model.", "0.") # Sequential এর ১ম অংশ ০
            if not k.startswith("projector"): # শুধু ক্লাসিফায়ার ওয়েটস নিচ্ছি
                new_sd[name] = v
        
        model.load_state_dict(new_sd, strict=False)
        model.eval()
        return model
    except Exception as e:
        return None

expert_model = load_expert_engine()

# ==========================================
# ৫. ড্যাশবোর্ড লজিক ও লগইন
# ==========================================
if 'user' not in st.session_state: st.session_state['user'] = None

with st.sidebar:
    st.title("🛡️ Secure Access")
    if st.session_state['user']:
        st.success(f"User: {st.session_state['user']}")
        if st.button("Logout"):
            st.session_state['user'] = None
            st.rerun()
    st.write("---")
    st.caption("Build v12.5 - SimCLR Optimized")

if not st.session_state['user']:
    st.markdown('<div class="glass-card"><h2>Expert Portal Login</h2></div>', unsafe_allow_html=True)
    user_input = st.text_input("Username")
    if st.button("Unlock Dashboard"):
        if user_input:
            st.session_state['user'] = user_input
            st.rerun()
else:
    st.markdown('<div class="glass-card"><h1>Expert Fish Analyzer</h1><p>High-Precision Neural Mapping Active</p></div>', unsafe_allow_html=True)
    
    file = st.file_uploader("Upload Fish Specimen", type=["jpg", "png", "jpeg"])
    
    if file:
        col1, col2 = st.columns([1, 1.2])
        img = Image.open(file).convert('RGB')
        
        with col1:
            st.image(img, caption="Target Specimen", use_container_width=True)
        
        with col2:
            if st.button("🚀 EXECUTE NEURAL ANALYSIS"):
                if expert_model:
                    with st.spinner("Decoding Neural Patterns..."):
                        # আপনার নোটবুকের Cell-2 অনুযায়ী ১৬০x১৬০ সাইজ
                        transform = transforms.Compose([
                            transforms.Resize((160, 160)),
                            transforms.ToTensor(),
                            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                        ])
                        
                        tensor = transform(img).unsqueeze(0)
                        
                        try:
                            with torch.no_grad():
                                output = expert_model(tensor)
                                prob = torch.nn.functional.softmax(output[0], dim=0)
                                conf, idx = torch.max(prob, 0)
                            
                            st.markdown(f'''
                                <div style="border: 2px solid #00C2FF; border-radius: 15px; padding: 25px; background: rgba(0,194,255,0.1);">
                                    <h2 style="color: #00C2FF; margin:0;">Identified: {CLASS_NAMES[idx.item()]}</h2>
                                    <h3 style="margin:0;">Precision: {conf.item()*100:.2f}%</h3>
                                </div>
                            ''', unsafe_allow_html=True)
                            
                            # টপ ৫ রেজাল্ট চার্ট
                            top5_p, top5_i = torch.topk(prob, 5)
                            df = pd.DataFrame({'Fish': [CLASS_NAMES[i] for i in top5_i], 'Match (%)': top5_p.numpy()*100})
                            st.bar_chart(df, x='Fish', y='Match (%)')
                        except Exception as e:
                            st.error("Analysis Failed. Model sync error.")
