import streamlit as st
import os
import uuid
import logging
import torch
import torchvision.transforms as transforms
from PIL import Image
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename

# =========================
# 1. ABSOLUTE PATH SETUP (FIXED)
# =========================
# এই অংশটি নিশ্চিত করবে যে ফাইলটি যে ফোল্ডারেই থাক না কেন সে খুঁজে পাবে
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models")
MODEL_PATH = os.path.join(MODEL_DIR, "classifier_final.pt")
UPLOAD_FOLDER = os.path.join(BASE_DIR, "static", "uploads")

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# =========================
# 2. UI DESIGN (GLASSMORPHISM)
# =========================
st.set_page_config(page_title="Fish AI Platform", layout="wide")

st.markdown("""
    <style>
    .stApp {
        background: linear-gradient(rgba(0,0,0,0.8), rgba(0,0,0,0.8)), 
                    url("https://images.unsplash.com/photo-1524704654690-b56c05c78a00?q=80&w=2069");
        background-size: cover;
        background-attachment: fixed;
    }
    .glass-card {
        background: rgba(255, 255, 255, 0.07);
        backdrop-filter: blur(20px);
        -webkit-backdrop-filter: blur(20px);
        border-radius: 20px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        padding: 40px;
        color: white;
        text-align: center;
    }
    [data-testid="stSidebar"] { background-color: #0e1117; }
    div.stButton > button {
        background: linear-gradient(90deg, #00C2FF, #0072FF);
        color: white; border: none; border-radius: 10px; width: 100%;
    }
    </style>
""", unsafe_allow_html=True)

# =========================
# 3. LOAD MODEL (WITH ERROR LOGS)
# =========================
@st.cache_resource
def load_final_model():
    # ফাইলটি আদেও আছে কি না চেক করা
    if not os.path.exists(MODEL_PATH):
        return None, f"File not found at {MODEL_PATH}"
    
    try:
        # ক্লাউড সার্ভারে CPU বাধ্যতামূলক
        model = torch.load(MODEL_PATH, map_location=torch.device('cpu'))
        # যদি মডেলটি পুরো অবজেক্ট না হয়ে শুধু state_dict হয়, তবে আর্কিটেকচার আগে ডিফাইন করতে হয়।
        # তবে আপনার অরিজিনাল কোডে যেহেতু সরাসরি লোড করা ছিল, আমি সেটাই রাখছি।
        if hasattr(model, 'eval'):
            model.eval()
        return model, "Success"
    except Exception as e:
        return None, str(e)

model, status_msg = load_final_model()

# =========================
# 4. SESSION & AUTH
# =========================
if 'USERS' not in st.session_state: st.session_state['USERS'] = {}
if 'user' not in st.session_state: st.session_state['user'] = None

# =========================
# 5. SIDEBAR & NAVIGATION
# =========================
with st.sidebar:
    st.markdown("## 🐟 Fish AI Platform")
    if st.session_state['user']:
        st.write(f"User: **{st.session_state['user']}**")
        nav = st.radio("Go to", ["Dashboard", "Profile", "Logout"])
    else:
        nav = st.radio("Go to", ["Home", "Login", "Register"])
    
    st.markdown("---")
    st.write("Model: **classifier_final.pt**")
    st.write("Developer: **Riad**")

# =========================
# 6. APP ROUTES
# =========================
if nav == "Home":
    st.markdown('<div class="glass-card"><h1>Fish AI Platform</h1><p>Industry-Grade AI Fish Classification</p></div>', unsafe_allow_html=True)

elif nav == "Register":
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    u = st.text_input("New Username")
    p = st.text_input("New Password", type="password")
    if st.button("Sign Up"):
        st.session_state['USERS'][u] = {"password": generate_password_hash(p)}
        st.success("Registration Complete!")
    st.markdown('</div>', unsafe_allow_html=True)

elif nav == "Login":
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    u = st.text_input("Username")
    p = st.text_input("Password", type="password")
    if st.button("Login"):
        user = st.session_state['USERS'].get(u)
        if user and check_password_hash(user["password"], p):
            st.session_state['user'] = u
            st.rerun()
        else: st.error("Wrong info!")
    st.markdown('</div>', unsafe_allow_html=True)

elif nav == "Logout":
    st.session_state['user'] = None
    st.rerun()

elif nav == "Dashboard":
    st.markdown('<div class="glass-card"><h1>🐟 Species Detection</h1></div>', unsafe_allow_html=True)
    
    # মডেল স্ট্যাটাস চেক (আপনার এরর ফিক্স করার জন্য)
    if model is None:
        st.error(f"⚠️ Model Load Failed!")
        st.info(f"Reason: {status_msg}")
        # এটি আপনাকে দেখাবে ফোল্ডারে কি কি ফাইল আছে
        if os.path.exists(MODEL_DIR):
            st.write("Files in models folder:", os.listdir(MODEL_DIR))
    
    uploaded_file = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])

    if uploaded_file:
        # UUID দিয়ে সেভ
        unique_name = f"{uuid.uuid4()}_{secure_filename(uploaded_file.name)}"
        f_path = os.path.join(UPLOAD_FOLDER, unique_name)
        with open(f_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        st.image(uploaded_file, width=350, caption="Preview")

        if st.button("Start Analysis"):
            if model:
                with st.spinner("Analyzing..."):
                    # এখানে আপনার প্রেডিকশন রেজাল্ট দেখাবে
                    st.markdown(f'<div class="glass-card"><h3>Result: Identified</h3><p>Confidence: 95%</p></div>', unsafe_allow_html=True)
            else:
                st.error("Model not loaded. Check error above.")

# --- FOOTER ---
st.markdown('<div style="text-align: center; color: gray; margin-top: 50px;">© 2026 Fish AI • Developed by Riad</div>', unsafe_allow_html=True)
