import streamlit as st
import os
import requests
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import pandas as pd
from werkzeug.security import generate_password_hash, check_password_hash

# ==========================================
# 1. CONFIG & HUGGING FACE PATH
# ==========================================
HF_MODEL_URL = "https://huggingface.co/riad300/fish-simclr-encoder/resolve/main/encoder_simclr.pt"
LOCAL_MODEL_PATH = "models/classifier_final.pt"
os.makedirs("models", exist_ok=True)

st.set_page_config(page_title="Fish AI - Global Enterprise", page_icon="🐟", layout="wide")

# ==========================================
# 2. UI THEME (FIXED CSS SYNTAX)
# ==========================================
def apply_pro_theme():
    # CSS এর ভেতরে ডাবল ব্র্যাকেট {{ }} ব্যবহার করা হয়েছে SyntaxError এড়াতে
    st.markdown("""
    <style>
    .stApp {{
        background: linear-gradient(rgba(0,0,0,0.85), rgba(0,0,0,0.85)), 
                    url("https://images.unsplash.com/photo-1524704654690-b56c05c78a00?q=80&w=2069");
        background-size: cover; background-attachment: fixed;
    }}
    .glass-panel {{
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(15px);
        border-radius: 20px; border: 1px solid rgba(255, 255, 255, 0.1);
        padding: 30px; margin-bottom: 20px; color: white; text-align: center;
    }}
    div.stButton > button {{
        background: linear-gradient(90deg, #00C2FF, #0072FF);
        color: white; border-radius: 10px; font-weight: bold; width: 100%; height: 3.5em;
        border: none;
    }}
    [data-testid="stSidebar"] {{
        background-color: #0e1117 !important;
    }}
    </style>
    """, unsafe_allow_html=True)

apply_pro_theme()

# ==========================================
# 3. AI ENGINE (Hugging Face)
# ==========================================
@st.cache_resource
def load_enterprise_model():
    if not os.path.exists(LOCAL_MODEL_PATH):
        try:
            response = requests.get(HF_MODEL_URL)
            with open(LOCAL_MODEL_PATH, "wb") as f:
                f.write(response.content)
        except: return None, "Hugging Face Connection Failed"
    
    try:
        model = models.resnet50(weights=None)
        model.fc = nn.Linear(model.fc.in_features, 21)
        state_dict = torch.load(LOCAL_MODEL_PATH, map_location=torch.device('cpu'))
        # SimCLR Key-Fixing
        new_state_dict = {k.replace("encoder.", ""): v for k, v in state_dict.items()}
        model.load_state_dict(new_state_dict, strict=False)
        model.eval()
        return model, "Operational (HF Cloud)"
    except Exception as e:
        return None, str(e)

model, model_status = load_enterprise_model()

CLASS_NAMES = [
    "Baim", "Bata", "Batasio(tenra)", "Chitul", "Croaker(Poya)", 
    "Hilsha", "Kajoli", "Meni", "Pabda", "Poli", "Puti", 
    "Rita", "Rui", "Rupchada", "Silver Carp", "Telapiya", 
    "carp", "k", "kaikka", "koral", "shrimp"
]

# ==========================================
# 4. AUTHENTICATION
# ==========================================
if 'USERS' not in st.session_state:
    st.session_state['USERS'] = {"admin": generate_password_hash("admin123")}
if 'user' not in st.session_state:
    st.session_state['user'] = None

# ==========================================
# 5. SIDEBAR & NAVIGATION
# ==========================================
with st.sidebar:
    st.title("🐟 Fish AI Pro")
    if st.session_state['user']:
        st.success(f"User: {st.session_state['user']}")
        menu = st.radio("Navigation", ["Dashboard", "Profile", "Logout"])
    else:
        menu = st.radio("Navigation", ["Login", "Register"])
    
    st.markdown("---")
    st.write(f"**AI Status:** {model_status}")
    st.write("Market Ready Build v2.1")

# ==========================================
# 6. PAGES
# ==========================================

if menu == "Login":
    st.markdown('<div class="glass-panel"><h2>Secure Access</h2></div>', unsafe_allow_html=True)
    u = st.text_input("Username")
    p = st.text_input("Password", type="password")
    if st.button("Login"):
        user_hash = st.session_state['USERS'].get(u)
        if user_hash and check_password_hash(user_hash, p):
            st.session_state['user'] = u
            st.rerun()
        else: st.error("Access Denied")

elif menu == "Register":
    st.markdown('<div class="glass-panel"><h2>Register Account</h2></div>', unsafe_allow_html=True)
    new_u = st.text_input("New Username")
    new_p = st.text_input("New Password", type="password")
    if st.button("Create"):
        st.session_state['USERS'][new_u] = generate_password_hash(new_p)
        st.success("Registration Successful")

elif menu == "Logout":
    st.session_state['user'] = None
    st.rerun()

elif menu == "Dashboard":
    if not st.session_state['user']:
        st.warning("Please Login First")
    else:
        st.markdown('<div class="glass-panel"><h1>Fish Identification Portal</h1></div>', unsafe_allow_html=True)
        
        file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])
        if file:
            col1, col2 = st.columns([1, 1.2])
            with col1:
                img = Image.open(file).convert('RGB')
                st.image(img, caption="Analyzed Specimen", use_container_width=True)
            
            with col2:
                if st.button("🚀 EXECUTE AI ANALYSIS"):
                    if model:
                        with st.spinner("Processing Neural Layers..."):
                            transform = transforms.Compose([
                                transforms.Resize((224, 224)),
                                transforms.ToTensor(),
                                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                            ])
                            input_t = transform(img).unsqueeze(0)
                            
                            with torch.no_grad():
                                output = model(input_t)
                                prob = torch.nn.functional.softmax(output[0], dim=0)
                                conf, idx = torch.max(prob, 0)
                            
                            st.markdown(f'''
                                <div class="glass-panel" style="border: 2px solid #00C2FF;">
                                    <h2 style="color: #00C2FF; margin-bottom: 0px;">Species: {CLASS_NAMES[idx.item()]}</h2>
                                    <h3>Confidence: {conf.item()*100:.2f}%</h3>
                                </div>
                            ''', unsafe_allow_html=True)
                            
                            # Probability Graph
                            top5_p, top5_i = torch.topk(prob, 5)
                            chart_df = pd.DataFrame({
                                'Species': [CLASS_NAMES[i] for i in top5_i],
                                'Confidence (%)': top5_p.numpy() * 100
                            })
                            st.write("#### Confidence Breakdown (Top 5)")
                            st.bar_chart(chart_df, x='Species', y='Confidence (%)', horizontal=True)

st.markdown('<p style="text-align:center; color:gray; margin-top:100px;">© 2026 Fish AI Enterprise • Developed by Riad</p>', unsafe_allow_html=True)
