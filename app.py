import streamlit as st
import os
import requests
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
from werkzeug.security import generate_password_hash, check_password_hash

# ==========================================
# 1. CONFIG & HUGGING FACE PATH
# ==========================================
# আপনার দেওয়া লিঙ্কে সরাসরি ফাইলটি পাওয়া যাবে
HF_MODEL_URL = "https://huggingface.co/riad300/fish-simclr-encoder/resolve/main/encoder_simclr.pt"
LOCAL_MODEL_PATH = "models/classifier_final.pt" 
os.makedirs("models", exist_ok=True)

st.set_page_config(page_title="Fish AI - Enterprise", page_icon="🐟", layout="wide")

# ==========================================
# 2. MODEL ENGINE (SMART LOADING)
# ==========================================
@st.cache_resource
def load_enterprise_model():
    # যদি লোকাল ফোল্ডারে না থাকে, তবে Hugging Face থেকে ডাউনলোড করবে
    if not os.path.exists(LOCAL_MODEL_PATH):
        with st.spinner("Downloading high-performance model from Hugging Face..."):
            response = requests.get(HF_MODEL_URL)
            with open(LOCAL_MODEL_PATH, "wb") as f:
                f.write(response.content)
    
    try:
        # ResNet50 Architecture for 21 Classes
        model = models.resnet50(weights=None)
        model.fc = nn.Linear(model.fc.in_features, 21)
        
        state_dict = torch.load(LOCAL_MODEL_PATH, map_location=torch.device('cpu'))
        
        # SimCLR Key-Fixing
        new_state_dict = {k.replace("encoder.", ""): v for k, v in state_dict.items()}
        
        model.load_state_dict(new_state_dict, strict=False)
        model.eval()
        return model, "Connected to Hugging Face"
    except Exception as e:
        return None, str(e)

model, status = load_enterprise_model()

# আপনার ২১টি ফিশ ক্লাস
CLASS_NAMES = [
    "Baim", "Bata", "Batasio(tenra)", "Chitul", "Croaker(Poya)", 
    "Hilsha", "Kajoli", "Meni", "Pabda", "Poli", "Puti", 
    "Rita", "Rui", "Rupchada", "Silver Carp", "Telapiya", 
    "carp", "k", "kaikka", "koral", "shrimp"
]

# ==========================================
# 3. AUTH & UI (MARKET READY)
# ==========================================
st.markdown("""
    <style>
    .stApp {
        background: linear-gradient(rgba(0,0,0,0.8), rgba(0,0,0,0.8)), 
                    url("https://images.unsplash.com/photo-1524704654690-b56c05c78a00?q=80&w=2069");
        background-size: cover; background-attachment: fixed;
    }
    .glass {
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(15px);
        border-radius: 20px; border: 1px solid rgba(255, 255, 255, 0.1);
        padding: 30px; margin-bottom: 20px; color: white; text-align: center;
    }
    div.stButton > button {
        background: linear-gradient(90deg, #00C2FF, #0072FF);
        color: white; border-radius: 10px; font-weight: bold; width: 100%;
    }
    </style>
""", unsafe_allow_html=True)

if 'user' not in st.session_state: st.session_state['user'] = None

# Sidebar
with st.sidebar:
    st.title("🐟 Fish AI Platform")
    if st.session_state['user']:
        st.success(f"User: {st.session_state['user']}")
        nav = st.radio("Menu", ["Dashboard", "Logout"])
    else:
        nav = st.radio("Menu", ["Login", "Register"])
    st.markdown("---")
    st.write(f"**AI Engine:** {status}")
    st.write("Developer: **Riad**")

# Pages
if nav == "Login":
    st.markdown('<div class="glass"><h2>Login</h2></div>', unsafe_allow_html=True)
    u = st.text_input("Username")
    if st.button("Enter"):
        st.session_state['user'] = u
        st.rerun()

elif nav == "Dashboard" and st.session_state['user']:
    st.markdown('<div class="glass"><h1>Fish Species Detection</h1><p>Hugging Face Integrated Build</p></div>', unsafe_allow_html=True)
    
    file = st.file_uploader("Upload Fish Image", type=["jpg", "png", "jpeg"])
    if file:
        col1, col2 = st.columns(2)
        with col1:
            st.image(file, caption="Input Image", use_container_width=True)
        with col2:
            if st.button("RUN AI ANALYSIS"):
                if model:
                    with st.spinner("Neural Processing..."):
                        # Transform
                        transform = transforms.Compose([
                            transforms.Resize((224, 224)),
                            transforms.ToTensor(),
                            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                        ])
                        img = Image.open(file).convert('RGB')
                        input_tensor = transform(img).unsqueeze(0)
                        
                        with torch.no_grad():
                            output = model(input_tensor)
                            prob = torch.nn.functional.softmax(output[0], dim=0)
                            conf, idx = torch.max(prob, 0)
                        
                        res = CLASS_NAMES[idx.item()]
                        st.markdown(f'''
                            <div class="glass" style="border: 2px solid #00C2FF;">
                                <h2 style="color: #00C2FF;">Result: {res}</h2>
                                <h3>Confidence: {conf.item()*100:.2f}%</h3>
                            </div>
                        ''', unsafe_allow_html=True)

elif nav == "Logout":
    st.session_state['user'] = None
    st.rerun()

st.markdown('<p style="text-align:center; color:gray; margin-top:100px;">© 2026 Fish AI | Enterprise Release</p>', unsafe_allow_html=True)
