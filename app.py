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
# 2. UI THEME (GLASSMORPHISM)
# ==========================================
def apply_pro_theme():
    st.markdown(f"""
    <style>
    .stApp {{
        background: linear-gradient(rgba(0,0,0,0.85), rgba(0,0,0,0.85)), 
                    url("https://images.unsplash.com/photo-1524704654690-b56c05c78a00?q=80&w=2069");
        background-size: cover; background-attachment: fixed;
    }
    .glass-panel {{
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(15px);
        border-radius: 20px; border: 1px solid rgba(255, 255, 255, 0.1);
        padding: 30px; margin-bottom: 20px; color: white;
    }
    div.stButton > button {{
        background: linear-gradient(90deg, #00C2FF, #0072FF);
        color: white; border-radius: 10px; font-weight: bold; width: 100%; height: 3.5em;
    }
    </style>
    """, unsafe_allow_html=True)

apply_pro_theme()

# ==========================================
# 3. AI ENGINE
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
        new_state_dict = {k.replace("encoder.", ""): v for k, v in state_dict.items()}
        model.load_state_dict(new_state_dict, strict=False)
        model.eval()
        return model, "Operational"
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
# 4. AUTHENTICATION SYSTEM
# ==========================================
if 'USERS' not in st.session_state:
    st.session_state['USERS'] = {"admin": generate_password_hash("admin123")}
if 'user' not in st.session_state:
    st.session_state['user'] = None

# ==========================================
# 5. NAVIGATION & SIDEBAR
# ==========================================
with st.sidebar:
    st.title("🐟 Fish AI Pro")
    if st.session_state['user']:
        st.success(f"User: {st.session_state['user']}")
        menu = st.radio("Navigation", ["Dashboard", "Analytics", "Logout"])
    else:
        menu = st.radio("Navigation", ["Login", "Register"])
    
    st.markdown("---")
    st.write(f"**Engine:** {model_status}")
    st.write("Built for **Industrial Fisheries**")

# ==========================================
# 6. APP PAGES
# ==========================================

# --- Login & Register ---
if menu == "Login":
    st.markdown('<div class="glass-panel"><h2>Secure Portal Access</h2></div>', unsafe_allow_html=True)
    u = st.text_input("Username")
    p = st.text_input("Password", type="password")
    if st.button("Enter Dashboard"):
        user_hash = st.session_state['USERS'].get(u)
        if user_hash and check_password_hash(user_hash, p):
            st.session_state['user'] = u
            st.rerun()
        else: st.error("Access Denied!")

elif menu == "Register":
    st.markdown('<div class="glass-panel"><h2>New Account</h2></div>', unsafe_allow_html=True)
    new_u = st.text_input("Username")
    new_p = st.text_input("Password", type="password")
    if st.button("Create Account"):
        st.session_state['USERS'][new_u] = generate_password_hash(new_p)
        st.success("Registration Successful!")

elif menu == "Logout":
    st.session_state['user'] = None
    st.rerun()

# --- Dashboard ---
elif menu == "Dashboard":
    st.markdown('<div class="glass-panel"><h1>Fish Species Detection</h1></div>', unsafe_allow_html=True)
    
    file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])
    if file:
        col1, col2 = st.columns([1, 1])
        with col1:
            img = Image.open(file).convert('RGB')
            st.image(img, caption="Original Input", use_container_width=True)
        
        with col2:
            if st.button("🚀 EXECUTE AI DIAGNOSTICS"):
                if model:
                    with st.spinner("Analyzing Morphology..."):
                        # Transform & Predict
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
                        
                        # Results
                        st.markdown(f'''
                            <div class="glass-panel" style="border: 2px solid #00C2FF;">
                                <h2 style="color: #00C2FF; margin-bottom: 0px;">Species: {CLASS_NAMES[idx.item()]}</h2>
                                <h3>Confidence: {conf.item()*100:.2f}%</h3>
                            </div>
                        ''', unsafe_allow_html=True)
                        
                        # Probability Chart
                        top5_p, top5_i = torch.topk(prob, 5)
                        chart_data = pd.DataFrame({
                            'Species': [CLASS_NAMES[i] for i in top5_i],
                            'Probability': top5_p.numpy() * 100
                        })
                        st.write("#### Top 5 Potential Matches")
                        st.bar_chart(chart_data, x='Species', y='Probability', horizontal=True)

# --- Analytics Page ---
elif menu == "Analytics":
    st.markdown('<div class="glass-panel"><h2>Market Insights & Model Stats</h2></div>', unsafe_allow_html=True)
    st.write("এখানে আপনি আপনার প্রজেক্টের স্ট্যাটিস্টিকস বা ডাটা হিস্ট্রি দেখাতে পারবেন।")

st.markdown('<p style="text-align:center; color:gray; margin-top:100px;">© 2026 Fish AI Enterprise • Verified Build</p>', unsafe_allow_html=True)
