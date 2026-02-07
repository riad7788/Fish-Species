import streamlit as st
import os
import uuid
import logging
import torch
from PIL import Image
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename

# =========================
# 1. CONFIG & PATHS (FIXED)
# =========================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, "static", "uploads")
MODEL_FOLDER = os.path.join(BASE_DIR, "models")
MODEL_PATH = os.path.join(MODEL_FOLDER, "classifier_final.pt")

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")

# Page Setup
st.set_page_config(page_title="Fish AI Platform", layout="wide")

# =========================
# 2. CUSTOM CSS (GLASSMORPHISM)
# =========================
def local_css():
    st.markdown(f"""
    <style>
    .stApp {{
        background: linear-gradient(rgba(0,0,0,0.7), rgba(0,0,0,0.7)), 
                    url("https://images.unsplash.com/photo-1524704654690-b56c05c78a00?q=80&w=2069");
        background-size: cover;
        background-attachment: fixed;
    }}
    .glass-card {{
        background: rgba(255, 255, 255, 0.08);
        backdrop-filter: blur(15px);
        -webkit-backdrop-filter: blur(15px);
        border-radius: 20px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        padding: 40px;
        margin: 10px auto;
        text-align: center;
        color: white;
    }}
    [data-testid="stSidebar"] {{
        background-color: #11141c !important;
    }}
    div.stButton > button {{
        background: linear-gradient(90deg, #00C2FF, #0072FF);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 10px 20px;
        font-weight: bold;
    }}
    </style>
    """, unsafe_allow_html=True)

local_css()

# =========================
# 3. SESSION STATE
# =========================
if 'USERS' not in st.session_state: st.session_state['USERS'] = {}
if 'user' not in st.session_state: st.session_state['user'] = None

# =========================
# 4. LOAD MODEL (FIXED LOGIC)
# =========================
@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        # এটি আপনাকে সাহায্য করবে বুঝতে যে আসলে পাথটি কোথায় খুঁজছে
        logging.error(f"Model not found at: {MODEL_PATH}")
        return None
    try:
        model = torch.load(MODEL_PATH, map_location="cpu")
        model.eval()
        logging.info("Model loaded successfully!")
        return model
    except Exception as e:
        logging.error(f"Error loading model: {e}")
        return None

model = load_model()

# =========================
# 5. SIDEBAR NAVIGATION
# =========================
with st.sidebar:
    st.markdown("## 🐟 Fish AI Platform")
    st.selectbox("Language", ["English", "Bengali"])
    
    if st.session_state['user']:
        st.success(f"Logged in: {st.session_state['user']}")
        choice = st.radio("Navigate", ["Dashboard", "Profile", "Logout"])
    else:
        choice = st.radio("Navigate", ["Home", "Login", "Register"])
    
    st.markdown("---")
    st.markdown("**Model Details**")
    st.write("• SimCLR (Self-Supervised)\n• ResNet50 Encoder\n• Linear Evaluation")
    
    st.markdown("---")
    st.markdown("Developer: **Riad**")

# =========================
# 6. APPLICATION LOGIC
# =========================

# --- HOME ---
if choice == "Home":
    st.markdown('<div class="glass-card"><h1>Welcome to Fish AI</h1><p>Industry-Grade Fish Classification Platform</p></div>', unsafe_allow_html=True)

# --- REGISTER ---
elif choice == "Register":
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.subheader("Create New Account")
    reg_u = st.text_input("Username", key="reg_u")
    reg_p = st.text_input("Password", type="password", key="reg_p")
    if st.button("Sign Up"):
        if reg_u in st.session_state['USERS']:
            st.error("User already exists!")
        elif reg_u and reg_p:
            st.session_state['USERS'][reg_u] = {"password": generate_password_hash(reg_p)}
            st.success("Account created! Please Login.")
            logging.info(f"New User: {reg_u}")
        else: st.warning("Fields cannot be empty.")
    st.markdown('</div>', unsafe_allow_html=True)

# --- LOGIN ---
elif choice == "Login":
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.subheader("Login")
    u = st.text_input("Username", key="log_u")
    p = st.text_input("Password", type="password", key="log_p")
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
    st.markdown(f'<div class="glass-card"><h2>Profile Details</h2><p>Username: <b>{st.session_state["user"]}</b></p><p>Status: Active</p></div>', unsafe_allow_html=True)

# --- DASHBOARD (THE MAIN FEATURE) ---
elif choice == "Dashboard":
    # হেডার কার্ড
    st.markdown('<div class="glass-card"><h1>🐟 Fish Species Detection</h1><p>Upload a fish image for AI analysis</p></div>', unsafe_allow_html=True)
    
    # মডেল চেক এরর হ্যান্ডলিং (ছবির মতো)
    if model is None:
        st.error(f"❌ Model (classifier_final.pt) not found in /models folder!")
        st.info(f"Current Path Checked: {MODEL_PATH}")

    # ফাইল আপলোডার
    file = st.file_uploader("", type=["jpg", "png", "jpeg"])
    
    if file:
        # ছবি সেভ করা (UUID লজিক)
        unique_id = str(uuid.uuid4())
        fname = secure_filename(file.name)
        save_path = os.path.join(UPLOAD_FOLDER, f"{unique_id}_{fname}")
        
        with open(save_path, "wb") as f:
            f.write(file.getbuffer())
        
        st.image(file, width=400, caption="Uploaded Image")
        
        if st.button("Analyze & Detect"):
            if model:
                with st.spinner("Analyzing Species..."):
                    # ডামি প্রেডিকশন
                    res_class = "Salmon" # আপনার মডেল থেকে আসবে
                    conf = 0.98
                    
                    st.markdown(f"""
                    <div class="glass-card" style="padding: 20px; border: 1px solid #00C2FF;">
                        <h2 style="color: #00C2FF;">Prediction: {res_class}</h2>
                        <h3>Confidence: {conf*100:.2f}%</h3>
                    </div>
                    """, unsafe_allow_html=True)
                    logging.info(f"Prediction by {st.session_state['user']}: {res_class}")
            else:
                st.error("Cannot perform detection without the model file.")

# --- FOOTER ---
st.markdown("""
<div style="text-align: center; color: rgba(255,255,255,0.4); font-size: 12px; margin-top: 50px; padding-bottom: 20px;">
    © 2026 • Fish AI Classification Platform<br>
    Built with PyTorch • SimCLR • Streamlit • Developed by Riad
</div>
""", unsafe_allow_html=True)
