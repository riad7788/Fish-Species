import streamlit as st
import os
import uuid
import logging
import torch
from PIL import Image
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename

# =========================
# 1. CONFIG & ABSOLUTE PATHS
# =========================
# এই অংশটি ক্লাউড সার্ভারে সঠিক পাথ খুঁজে পেতে সাহায্য করবে
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# মডেল পাথে কোনো ভুল রাখা যাবে না
MODEL_PATH = os.path.join(BASE_DIR, "models", "classifier_final.pt")
UPLOAD_FOLDER = os.path.join(BASE_DIR, "static", "uploads")

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")

# Page Layout
st.set_page_config(page_title="Fish AI Platform", layout="wide")

# =========================
# 2. UI DESIGN (GLASSMORPHISM)
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
# 4. LOAD MODEL (CACHED)
# =========================
@st.cache_resource
def load_fish_model():
    # এখানে পাথটি আবার ভেরিফাই করছি
    if not os.path.isfile(MODEL_PATH):
        return None
    try:
        # ম্যাপ লোকেশন CPU দেওয়া হয়েছে যেন সার্ভারে জিপিইউ না থাকলেও কাজ করে
        model = torch.load(MODEL_PATH, map_location=torch.device('cpu'))
        model.eval()
        return model
    except Exception as e:
        logging.error(f"Error loading model: {e}")
        return None

# মডেল লোড করার চেষ্টা
model = load_fish_model()

# =========================
# 5. SIDEBAR NAVIGATION
# =========================
with st.sidebar:
    st.markdown("## 🐟 Fish AI Platform")
    if st.session_state['user']:
        st.write(f"Logged in as: **{st.session_state['user']}**")
        choice = st.radio("Navigate", ["Dashboard", "Profile", "Logout"])
    else:
        choice = st.radio("Navigate", ["Home", "Login", "Register"])
    
    st.markdown("---")
    st.markdown("**Model Specs**")
    st.write("• ResNet50\n• SimCLR V2\n• PyTorch")
    st.markdown("---")
    st.write("Developer: **Riad**")

# =========================
# 6. APP LOGIC
# =========================

if choice == "Home":
    st.markdown('<div class="glass-card"><h1>Welcome to Fish AI</h1><p>Next-Gen Fisheries Analysis</p></div>', unsafe_allow_html=True)

elif choice == "Register":
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.subheader("Account Registration")
    reg_u = st.text_input("New Username")
    reg_p = st.text_input("New Password", type="password")
    if st.button("Sign Up"):
        if reg_u and reg_p:
            st.session_state['USERS'][reg_u] = {"password": generate_password_hash(reg_p)}
            st.success("Registered successfully! Go to Login.")
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
        else: st.error("Wrong info!")
    st.markdown('</div>', unsafe_allow_html=True)

elif choice == "Logout":
    st.session_state['user'] = None
    st.rerun()

elif choice == "Profile":
    st.markdown(f'<div class="glass-card"><h2>User Profile</h2><p>Account: {st.session_state["user"]}</p></div>', unsafe_allow_html=True)

elif choice == "Dashboard":
    st.markdown('<div class="glass-card"><h1>🐟 Fish Species Detection</h1><p>Industry-Grade AI Fish Classification Platform</p></div>', unsafe_allow_html=True)
    
    # এরর মেসেজ যদি মডেল না পাওয়া যায়
    if model is None:
        st.error(f"Model file 'classifier_final.pt' not found in /models folder!")
        st.info(f"Checking path: {MODEL_PATH}")

    # ড্যাশবোর্ড কন্টেন্ট
    uploaded_file = st.file_uploader("", type=["jpg", "png", "jpeg"])

    if uploaded_file:
        # অরিজিনাল Flask লজিক: UUID দিয়ে ফাইল সেভ
        unique_name = f"{uuid.uuid4()}_{secure_filename(uploaded_file.name)}"
        full_save_path = os.path.join(UPLOAD_FOLDER, unique_name)
        
        with open(full_save_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        st.image(uploaded_file, width=350, caption="Uploaded Image Preview")

        if st.button("Analyze & Detect"):
            if model:
                with st.spinner("Classifying..."):
                    # Dummy results (Replace with model(input) logic)
                    res = "Species: Lates calcarifer"
                    conf = "94.8%"
                    
                    st.markdown(f"""
                    <div class="glass-card" style="padding: 20px; border-left: 5px solid #00C2FF;">
                        <h3 style="color: #00C2FF;">{res}</h3>
                        <p>Confidence Score: {conf}</p>
                    </div>
                    """, unsafe_allow_html=True)
                    logging.info(f"User {st.session_state['user']} analyzed {unique_name}")
            else:
                st.error("Operation failed. Model not loaded.")

# --- FOOTER ---
st.markdown("""
<div style="text-align: center; color: rgba(255,255,255,0.4); font-size: 11px; margin-top: 50px;">
    © 2026 • Fish AI Classification Platform<br>
    Built with PyTorch • SimCLR • Streamlit • Developed by Riad
</div>
""", unsafe_allow_html=True)
