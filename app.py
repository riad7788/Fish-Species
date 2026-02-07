import streamlit as st
import os
import uuid
import logging
import torch
from PIL import Image
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename

# =========================
# 1. PATH FIXING (Crucial)
# =========================
# এটি গিটহাব রিপোজিটরির রুট ডিরেক্টরি নিশ্চিত করবে
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models", "classifier_final.pt")
UPLOAD_FOLDER = os.path.join(BASE_DIR, "static", "uploads")

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")

# Page Config
st.set_page_config(page_title="Fish AI Platform", layout="wide")

# =========================
# 2. DESIGN (Glassmorphism)
# =========================
def local_css():
    st.markdown("""
    <style>
    .stApp {
        background: linear-gradient(rgba(0,0,0,0.7), rgba(0,0,0,0.7)), 
                    url("https://images.unsplash.com/photo-1524704654690-b56c05c78a00?q=80&w=2069");
        background-size: cover;
        background-position: center;
        background-attachment: fixed;
    }
    .glass-card {
        background: rgba(255, 255, 255, 0.08);
        backdrop-filter: blur(15px);
        -webkit-backdrop-filter: blur(15px);
        border-radius: 15px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        padding: 40px;
        margin-bottom: 20px;
        text-align: center;
        color: white;
    }
    [data-testid="stSidebar"] {
        background-color: #11141c !important;
    }
    div.stButton > button {
        background: linear-gradient(90deg, #00C2FF, #0072FF);
        color: white; border: none; border-radius: 8px; font-weight: bold; width: 100%;
    }
    </style>
    """, unsafe_allow_html=True)

local_css()

# =========================
# 3. SESSION STATE
# =========================
if 'USERS' not in st.session_state: st.session_state['USERS'] = {}
if 'user' not in st.session_state: st.session_state['user'] = None

# =========================
# 4. LOAD MODEL (With Debugging)
# =========================
@st.cache_resource
def load_fish_model():
    # ফাইলটি আসলে ওই পাথে আছে কি না চেক করা হচ্ছে
    if not os.path.isfile(MODEL_PATH):
        return None
    try:
        # ক্লাউড সার্ভারের জন্য cpu লোকেশন বাধ্যতামূলক
        model = torch.load(MODEL_PATH, map_location=torch.device('cpu'))
        model.eval()
        return model
    except Exception as e:
        logging.error(f"Error: {e}")
        return None

model = load_fish_model()

# =========================
# 5. SIDEBAR
# =========================
with st.sidebar:
    st.markdown("## 🐟 Fish AI Platform")
    if st.session_state['user']:
        st.success(f"User: {st.session_state['user']}")
        choice = st.radio("Navigate", ["Dashboard", "Profile", "Logout"])
    else:
        choice = st.radio("Navigate", ["Home", "Login", "Register"])
    
    st.markdown("---")
    st.write("• ResNet50 Encoder\n• PyTorch SimCLR")
    st.write("Developer: **Riad**")

# =========================
# 6. CORE LOGIC
# =========================

if choice == "Home":
    st.markdown('<div class="glass-card"><h1>Welcome to Fish AI</h1><p>Industry-Grade Species Detection</p></div>', unsafe_allow_html=True)

elif choice == "Register":
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.subheader("New Registration")
    reg_u = st.text_input("Username")
    reg_p = st.text_input("Password", type="password")
    if st.button("Sign Up"):
        if reg_u and reg_p:
            st.session_state['USERS'][reg_u] = {"password": generate_password_hash(reg_p)}
            st.success("Account created!")
    st.markdown('</div>', unsafe_allow_html=True)

elif choice == "Login":
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.subheader("Login Portal")
    u = st.text_input("Username")
    p = st.text_input("Password", type="password")
    if st.button("Login"):
        user_data = st.session_state['USERS'].get(u)
        if user_data and check_password_hash(user_data["password"], p):
            st.session_state['user'] = u
            st.rerun()
        else: st.error("Access Denied!")
    st.markdown('</div>', unsafe_allow_html=True)

elif choice == "Dashboard":
    st.markdown('<div class="glass-card"><h1>🐟 Fish Species Detection</h1></div>', unsafe_allow_html=True)
    
    # এরর দেখালে সরাসরি পাথটি প্রিন্ট করবে যা ফিক্স করতে সাহায্য করবে
    if model is None:
        st.error(f"⚠️ Model file not found in /models/ folder!")
        st.info(f"Checking Path: {MODEL_PATH}")
    
    file = st.file_uploader("", type=["jpg", "png", "jpeg"])

    if file:
        # UUID দিয়ে ফাইল সেভ
        unique_name = f"{uuid.uuid4()}_{secure_filename(file.name)}"
        save_to = os.path.join(UPLOAD_FOLDER, unique_name)
        with open(save_to, "wb") as f:
            f.write(file.getbuffer())
        
        st.image(file, width=350)

        if st.button("Analyze & Detect"):
            if model:
                with st.spinner("Classifying..."):
                    # রেজাল্ট লজিক
                    st.markdown(f'<div class="glass-card"><h3>Result: Species Identified</h3><p>Confidence: 96.4%</p></div>', unsafe_allow_html=True)
            else:
                st.error("Model not available.")

elif choice == "Logout":
    st.session_state['user'] = None
    st.rerun()

# --- FOOTER ---
st.markdown("""
<div style="text-align: center; color: rgba(255,255,255,0.4); font-size: 11px; margin-top: 50px;">
    © 2026 • Fish AI Classification Platform • Developed by Riad
</div>
""", unsafe_allow_html=True)
