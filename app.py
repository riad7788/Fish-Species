import streamlit as st
import os
import uuid
import logging
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
from werkzeug.security import generate_password_hash, check_password_hash

# ==========================================
# 1. DIRECTORY & LOGGING SETUP
# ==========================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models", "classifier_final.pt")
UPLOAD_DIR = os.path.join(BASE_DIR, "static", "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")

# Page Config
st.set_page_config(page_title="Fish AI Pro", page_icon="🐟", layout="wide")

# ==========================================
# 2. MODEL ARCHITECTURE (FIXES THE 'OrderedDict' ERROR)
# ==========================================
def get_fish_model(num_classes):
    # ইন্ডাস্ট্রি স্ট্যান্ডার্ড: ResNet50 ব্যাকবোন ব্যবহার
    model = models.resnet50(weights=None) 
    # আপনার মডেলের শেষ লেয়ারটি আপনার ক্লাসের সংখ্যা (২১) অনুযায়ী সেট করা
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)
    return model

@st.cache_resource
def load_trained_model():
    if not os.path.exists(MODEL_PATH):
        return None, f"Model missing at {MODEL_PATH}"
    
    try:
        # ২১টি ক্লাসের জন্য মডেল স্ট্রাকচার তৈরি
        model = get_fish_model(num_classes=21)
        
        # ওয়েটস লোড করা (Fixes the OrderedDict error)
        state_dict = torch.load(MODEL_PATH, map_location=torch.device('cpu'))
        model.load_state_dict(state_dict)
        model.eval()
        return model, "Success"
    except Exception as e:
        return None, str(e)

# ==========================================
# 3. UI STYLING (GLASSMORPHISM)
# ==========================================
st.markdown("""
    <style>
    .stApp {
        background: linear-gradient(rgba(0,0,0,0.8), rgba(0,0,0,0.8)), 
                    url("https://images.unsplash.com/photo-1524704654690-b56c05c78a00?q=80&w=2069");
        background-size: cover; background-attachment: fixed;
    }
    .glass-card {
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(15px);
        border-radius: 20px; border: 1px solid rgba(255, 255, 255, 0.1);
        padding: 30px; margin-bottom: 20px; color: white; text-align: center;
    }
    div.stButton > button {
        background: linear-gradient(90deg, #00C2FF, #0072FF);
        color: white; border: none; border-radius: 10px; font-weight: bold; width: 100%; height: 3em;
    }
    </style>
""", unsafe_allow_html=True)

# ==========================================
# 4. CLASSES & PREDICTION LOGIC
# ==========================================
CLASS_NAMES = [
    "Baim", "Bata", "Batasio(tenra)", "Chitul", "Croaker(Poya)", 
    "Hilsha", "Kajoli", "Meni", "Pabda", "Poli", "Puti", 
    "Rita", "Rui", "Rupchada", "Silver Carp", "Telapiya", 
    "carp", "k", "kaikka", "koral", "shrimp"
]

def predict_fish(model, image_file):
    # ইন্ডাস্ট্রি স্ট্যান্ডার্ড ইমেজ ট্রান্সফর্ম
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    image = Image.open(image_file).convert('RGB')
    input_tensor = transform(image).unsqueeze(0)
    
    with torch.no_grad():
        outputs = model(input_tensor)
        probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
        confidence, index = torch.max(probabilities, 0)
    
    return CLASS_NAMES[index], confidence.item()

# ==========================================
# 5. MAIN APP FLOW
# ==========================================
model, status = load_trained_model()

# Session State
if 'user' not in st.session_state: st.session_state['user'] = None

with st.sidebar:
    st.title("🐟 Fish AI Platform")
    if st.session_state['user']:
        st.write(f"Logged in as: **{st.session_state['user']}**")
        if st.button("Logout"):
            st.session_state['user'] = None
            st.rerun()
    else:
        st.info("Please Login to access Dashboard")

# Dashboard Logic
if not st.session_state['user']:
    st.markdown('<div class="glass-card"><h2>Welcome</h2><p>Please enter your name to start</p></div>', unsafe_allow_html=True)
    user_input = st.text_input("Username")
    if st.button("Login"):
        st.session_state['user'] = user_input
        st.rerun()
else:
    st.markdown('<div class="glass-card"><h1>Fish Species Detection</h1></div>', unsafe_allow_html=True)
    
    if model is None:
        st.error(f"System Error: {status}")
    
    file = st.file_uploader("Upload Fish Image", type=["jpg", "png", "jpeg"])
    
    if file:
        st.image(file, width=400, caption="Uploaded Image")
        if st.button("Run AI Analysis"):
            if model:
                with st.spinner("Analyzing Morphology..."):
                    try:
                        name, conf = predict_fish(model, file)
                        st.markdown(f'''
                            <div class="glass-card" style="border: 2px solid #00C2FF;">
                                <h2 style="color: #00C2FF;">Result: {name}</h2>
                                <h3>Confidence: {conf*100:.2f}%</h3>
                            </div>
                        ''', unsafe_allow_html=True)
                    except Exception as e:
                        st.error(f"Analysis Failed: {e}")

st.markdown('<p style="text-align:center; color:gray; font-size:12px; margin-top:50px;">© 2026 Fish AI • Developed by Riad</p>', unsafe_allow_html=True)
