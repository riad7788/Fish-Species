import streamlit as st
import os
import uuid
import logging
import torch
from PIL import Image, ImageDraw, ImageFont
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename

# =========================
# 1. BASIC CONFIG & LOGGING
# =========================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, "static/uploads")
MODEL_FOLDER = os.path.join(BASE_DIR, "models")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

# Page Config
st.set_page_config(page_title="Fish AI Platform", layout="wide")

# =========================
# 2. SESSION STATE (Flask এর মত ডাটা ধরে রাখা)
# =========================
if 'USERS' not in st.session_state: st.session_state['USERS'] = {}
if 'user' not in st.session_state: st.session_state['user'] = None

# =========================
# 3. CUSTOM CSS (Design logic)
# =========================
def local_css():
    st.markdown("""
    <style>
    .stApp {
        background: linear-gradient(rgba(0,0,0,0.7), rgba(0,0,0,0.7)), 
                    url("https://images.unsplash.com/photo-1524704654690-b56c05c78a00?q=80&w=2069");
        background-size: cover;
        background-attachment: fixed;
    }
    .glass-card {
        background: rgba(255, 255, 255, 0.08);
        backdrop-filter: blur(15px);
        border-radius: 20px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        padding: 40px;
        margin: 10px auto;
        text-align: center;
        color: white;
    }
    [data-testid="stSidebar"] { background-color: #11141c; }
    div.stButton > button {
        background: linear-gradient(90deg, #00C2FF, #0072FF);
        color: white; border: none; border-radius: 8px; width: 100%;
    }
    </style>
    """, unsafe_allow_html=True)

local_css()

# =========================
# 4. LOAD MODEL (classifier_final.pt)
# =========================
MODEL_PATH = os.path.join(MODEL_FOLDER, "classifier_final.pt")

@st.cache_resource
def load_model():
    if os.path.exists(MODEL_PATH):
        try:
            m = torch.load(MODEL_PATH, map_location="cpu")
            m.eval()
            logging.info("Model classifier_final.pt loaded successfully")
            return m
        except Exception as e:
            logging.error(f"Model load failed: {e}")
    return None

model = load_model()

# =========================
# 5. SIDEBAR (Navigation & Features)
# =========================
with st.sidebar:
    st.markdown("## 🐟 Fish AI Platform")
    if st.session_state['user']:
        st.success(f"Logged in: {st.session_state['user']}")
        choice = st.radio("Navigate", ["Dashboard", "Profile", "Logout"])
    else:
        choice = st.radio("Navigate", ["Home", "Login", "Register"])
    
    st.markdown("---")
    st.markdown("**Settings**")
    st.checkbox("Enable Explainability", value=True)
    st.checkbox("Enable PDF Report")
    st.markdown("---")
    st.write("Developed by Riad")

# =========================
# 6. APP LOGIC (Routes)
# =========================

# --- HOME ---
if choice == "Home":
    st.markdown('<div class="glass-card"><h1>Welcome to Fish AI</h1><p>Industry-Grade Classification</p></div>', unsafe_allow_html=True)

# --- REGISTER ---
elif choice == "Register":
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.subheader("Create Account")
    reg_u = st.text_input("Username")
    reg_p = st.text_input("Password", type="password")
    if st.button("Register Now"):
        if reg_u in st.session_state['USERS']: st.error("User exists!")
        else:
            st.session_state['USERS'][reg_u] = {"password": generate_password_hash(reg_p)}
            st.success("Registered! Go to Login.")
            logging.info(f"User Registered: {reg_u}")
    st.markdown('</div>', unsafe_allow_html=True)

# --- LOGIN ---
elif choice == "Login":
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.subheader("User Login")
    u = st.text_input("Username")
    p = st.text_input("Password", type="password")
    if st.button("Login"):
        user_data = st.session_state['USERS'].get(u)
        if user_data and check_password_hash(user_data["password"], p):
            st.session_state['user'] = u
            st.rerun()
        else: st.error("Invalid credentials")
    st.markdown('</div>', unsafe_allow_html=True)

# --- LOGOUT ---
elif choice == "Logout":
    st.session_state['user'] = None
    st.rerun()

# --- PROFILE ---
elif choice == "Profile":
    st.markdown(f'<div class="glass-card"><h2>Profile</h2><p>Username: {st.session_state["user"]}</p></div>', unsafe_allow_html=True)

# --- DASHBOARD (Main Functionality) ---
elif choice == "Dashboard":
    st.markdown('<div class="glass-card"><h1>🐟 Fish Species Detection</h1><p>Upload a fish image for AI analysis</p></div>', unsafe_allow_html=True)
    
    file = st.file_uploader("", type=["jpg", "png", "jpeg"])
    
    if file:
        # UUID & Secure Filename (আপনার অরিজিনাল লজিক)
        fname = secure_filename(file.name)
        unique_name = f"{uuid.uuid4()}_{fname}"
        path = os.path.join(UPLOAD_FOLDER, unique_name)
        
        with open(path, "wb") as f:
            f.write(file.getbuffer())
        
        st.image(file, width=400, caption="Uploaded Image")
        
        if st.button("Analyze & Detect"):
            if model:
                # Prediction Logic
                with st.spinner("Processing..."):
                    res_class = "Class A" # Dummy
                    conf = 0.92
                    
                    st.markdown(f"""
                    <div class="glass-card">
                        <h3>Prediction: {res_class}</h3>
                        <h4>Confidence: {conf*100}%</h4>
                    </div>
                    """, unsafe_allow_html=True)
                    logging.info(f"Prediction by {st.session_state['user']}: {res_class}")
            else:
                st.error("Model (classifier_final.pt) not found in /models folder!")

# --- FOOTER ---
st.markdown("""
<div style="text-align: center; color: rgba(255,255,255,0.4); font-size: 12px; margin-top: 50px;">
    © 2026 • Fish AI Classification Platform<br>
    Built with PyTorch • SimCLR • Streamlit • Developed by Riad
</div>
""", unsafe_allow_html=True)
