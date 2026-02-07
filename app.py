import streamlit as st
import os
import requests
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import pandas as pd # গ্রাফের জন্য

# --- PATH & SETUP ---
HF_MODEL_URL = "https://huggingface.co/riad300/fish-simclr-encoder/resolve/main/encoder_simclr.pt"
LOCAL_MODEL_PATH = "models/classifier_final.pt"
os.makedirs("models", exist_ok=True)

st.set_page_config(page_title="Fish AI - Market Pro", page_icon="🐟", layout="wide")

# --- MODEL ENGINE ---
@st.cache_resource
def load_final_model():
    if not os.path.exists(LOCAL_MODEL_PATH):
        response = requests.get(HF_MODEL_URL)
        with open(LOCAL_MODEL_PATH, "wb") as f:
            f.write(response.content)
    
    try:
        # ResNet50 for 21 Classes
        model = models.resnet50(weights=None)
        model.fc = nn.Linear(model.fc.in_features, 21)
        
        state_dict = torch.load(LOCAL_MODEL_PATH, map_location=torch.device('cpu'))
        # SimCLR Key-Fixing
        new_state_dict = {k.replace("encoder.", ""): v for k, v in state_dict.items()}
        
        model.load_state_dict(new_state_dict, strict=False)
        model.eval()
        return model
    except Exception as e:
        return str(e)

model = load_final_model()

# --- ২১টি ক্লাসের সঠিক লিস্ট ---
CLASS_NAMES = [
    "Baim", "Bata", "Batasio(tenra)", "Chitul", "Croaker(Poya)", 
    "Hilsha", "Kajoli", "Meni", "Pabda", "Poli", "Puti", 
    "Rita", "Rui", "Rupchada", "Silver Carp", "Telapiya", 
    "carp", "k", "kaikka", "koral", "shrimp"
]

# --- UI STYLING ---
st.markdown("""
    <style>
    .stApp { background: #0e1117; color: white; }
    .res-card {
        background: rgba(255, 255, 255, 0.05);
        padding: 20px; border-radius: 15px; border-left: 5px solid #00C2FF;
    }
    </style>
""", unsafe_allow_html=True)

# --- APP LAYOUT ---
st.title("🐟 Fish AI Industry Dashboard")

file = st.file_uploader("Upload Fish Image", type=["jpg", "png", "jpeg"])

if file:
    col1, col2 = st.columns([1, 1.2])
    
    with col1:
        img = Image.open(file).convert('RGB')
        st.image(img, caption="Input Image", use_container_width=True)
        
    with col2:
        if st.button("🚀 START DEEP ANALYSIS"):
            if isinstance(model, str):
                st.error(f"Model Error: {model}")
            else:
                with st.spinner("Processing Neural Layers..."):
                    # Pre-processing
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
                    
                    # রেজাল্ট ফিল্টারিং (ইন্ডাস্ট্রি স্ট্যান্ডার্ড)
                    if conf.item() < 0.40: # যদি ৪০% এর নিচে কনফিডেন্স হয়
                        st.warning("⚠️ Low Confidence! The image might be unclear or a new species.")
                    
                    # টপ ৫ প্রেডিকশন গ্রাফ (কেন ভুল হচ্ছে তা বুঝার জন্য)
                    top5_prob, top5_idx = torch.topk(prob, 5)
                    chart_data = pd.DataFrame({
                        'Species': [CLASS_NAMES[i] for i in top5_idx],
                        'Confidence': top5_prob.numpy() * 100
                    })
                    
                    st.markdown(f'''
                        <div class="res-card">
                            <h2 style="color:#00C2FF;">Species: {CLASS_NAMES[idx.item()]}</h2>
                            <h3>Match: {conf.item()*100:.2f}%</h3>
                        </div>
                    ''', unsafe_allow_html=True)
                    
                    st.write("### Analysis Breakdown (Top 5 Matches)")
                    st.bar_chart(chart_data, x='Species', y='Confidence', horizontal=True)

st.sidebar.info("System Build: HF-V2.0\nStatus: Operational")
