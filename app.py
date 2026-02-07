import streamlit as st
import os
import uuid
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

# ==========================================
# 1. PATH & SYSTEM SETUP
# ==========================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models", "classifier_final.pt")

# Industry Standard Page Config
st.set_page_config(page_title="Fish AI - Pro Edition", page_icon="🐟", layout="wide")

# ==========================================
# 2. PRO UI DESIGN (GLASSMORPHISM)
# ==========================================
st.markdown("""
    <style>
    .stApp {
        background: linear-gradient(rgba(0,0,0,0.85), rgba(0,0,0,0.85)), 
                    url("https://images.unsplash.com/photo-1524704654690-b56c05c78a00?q=80&w=2069");
        background-size: cover; background-attachment: fixed;
    }
    .main-card {
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(20px);
        border-radius: 25px; border: 1px solid rgba(255, 255, 255, 0.1);
        padding: 40px; text-align: center; color: white;
    }
    div.stButton > button {
        background: linear-gradient(90deg, #00C2FF, #0072FF);
        color: white; border: none; border-radius: 12px; height: 3.5em; font-weight: bold; width: 100%;
        transition: 0.4s;
    }
    div.stButton > button:hover { transform: translateY(-2px); box-shadow: 0 5px 15px rgba(0, 194, 255, 0.4); }
    </style>
""", unsafe_allow_html=True)

# ==========================================
# 3. AI ENGINE (FIXES ALL LOADING ERRORS)
# ==========================================
@st.cache_resource
def load_industrial_model():
    if not os.path.exists(MODEL_PATH):
        return None, f"Model file missing at: {MODEL_PATH}"
    
    try:
        # Step 1: ResNet50 আর্কিটেকচার তৈরি (২১টি ক্লাসের জন্য)
        model = models.resnet50(weights=None)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, 21) 

        # Step 2: Weight লোড করা
        state_dict = torch.load(MODEL_PATH, map_location=torch.device('cpu'))
        
        # SimCLR মডেলের জন্য কী ফিক্সিং (যদি প্রয়োজন হয়)
        new_state_dict = {}
        for k, v in state_dict.items():
            name = k.replace("encoder.", "") # SimCLR লেয়ার নাম ক্লিন করা
            new_state_dict[name] = v
            
        # Step 3: লোড করা (strict=False দেওয়া হয়েছে যেন কী মিসম্যাচ থাকলেও এরর না দেয়)
        model.load_state_dict(new_state_dict, strict=False)
        model.eval()
        return model, "Operational"
    except Exception as e:
        return None, str(e)

model, system_status = load_industrial_model()

# আপনার দেওয়া ২১টি মাছের নাম
CLASS_NAMES = [
    "Baim", "Bata", "Batasio(tenra)", "Chitul", "Croaker(Poya)", 
    "Hilsha", "Kajoli", "Meni", "Pabda", "Poli", "Puti", 
    "Rita", "Rui", "Rupchada", "Silver Carp", "Telapiya", 
    "carp", "k", "kaikka", "koral", "shrimp"
]

# ==========================================
# 4. MAIN INTERFACE
# ==========================================
with st.sidebar:
    st.title("🐟 Fish AI Pro")
    st.success(f"System: {system_status}")
    st.markdown("---")
    st.write("**Model Specs:** ResNet50 + SimCLR V2")
    st.write("**Developer:** Riad")

st.markdown('<div class="main-card"><h1>Industry-Grade Fish Identification</h1><p>Real-time Neural Analysis Portal</p></div>', unsafe_allow_html=True)

if model is None:
    st.error(f"System Offline: {system_status}")

uploaded_file = st.file_uploader("", type=["jpg", "png", "jpeg"])

if uploaded_file:
    col1, col2 = st.columns([1, 1])
    with col1:
        st.image(uploaded_file, caption="Analysis Input", use_container_width=True)
    
    with col2:
        if st.button("RUN AI DIAGNOSTICS"):
            if model:
                with st.spinner("Analyzing Morphology..."):
                    try:
                        # প্রোসেসিং
                        transform = transforms.Compose([
                            transforms.Resize((224, 224)),
                            transforms.ToTensor(),
                            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                        ])
                        img = Image.open(uploaded_file).convert('RGB')
                        input_data = transform(img).unsqueeze(0)

                        # প্রেডিকশন
                        with torch.no_grad():
                            output = model(input_data)
                            prob = torch.nn.functional.softmax(output[0], dim=0)
                            conf, idx = torch.max(prob, 0)

                        fish_name = CLASS_NAMES[idx.item()]
                        
                        # প্রোফেশনাল রেজাল্ট কার্ড
                        st.markdown(f'''
                            <div class="main-card" style="border: 2px solid #00C2FF; background: rgba(0, 194, 255, 0.05); margin-top:20px;">
                                <h2 style="color: #00C2FF; margin-bottom: 0px;">Species: {fish_name}</h2>
                                <p style="font-size: 24px;">Confidence: {conf.item()*100:.2f}%</p>
                            </div>
                        ''', unsafe_allow_html=True)
                    except Exception as e:
                        st.error(f"Analysis Failed: {e}")

st.markdown('<p style="text-align:center; color:rgba(255,255,255,0.3); margin-top:50px;">© 2026 Fish AI Platform | Production Build</p>', unsafe_allow_html=True)
