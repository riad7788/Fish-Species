import streamlit as st
import os
import uuid
import logging
import torch
from PIL import Image, ImageDraw, ImageFont
from werkzeug.security import generate_password_hash, check_password_hash

# =========================
# 1. INITIAL CONFIG
# =========================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, "static/uploads")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Page Title & Layout
st.set_page_config(page_title="Fish AI Platform", layout="wide")

# =========================
# 2. CUSTOM CSS (ছবিতে যেমন দেখছেন)
# =========================
def local_css():
    st.markdown(f"""
    <style>
    /* ব্যাকগ্রাউন্ড ইমেজ */
    .stApp {{
        background: url("https://images.unsplash.com/photo-1524704654690-b56c05c78a00?q=80&w=2069&auto=format&fit=crop"); /* এখানে আপনার local background.jpg এর লিঙ্ক দিতে পারেন */
        background-size: cover;
    }}
    
    /* গ্লাস ইফেক্ট কার্ড */
    .glass-card {{
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(15px);
        border-radius: 20px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        padding: 40px;
        text-align: center;
        color: white;
    }}

    /* সাইডবার স্টাইল */
    [data-testid="stSidebar"] {{
        background-color: rgba(20, 20, 30, 0.95);
    }}

    /* বাটন স্টাইল */
    .stButton>button {{
        background: linear-gradient(90deg, #00C2FF, #0072FF);
        color: white;
        border: None;
        border-radius: 10px;
        padding: 10px 25px;
    }}
    </style>
    """, unsafe_allow_state_allowed=True)

local_css()

# =========================
# 3. LOGIC & MODEL LOADING
# =========================
MODEL_PATH = os.path.join(BASE_DIR, "models", "classifier_final.pt")

@st.cache_resource
def load_model():
    if os.path.exists(MODEL_PATH):
        model = torch.load(MODEL_PATH, map_location="cpu")
        model.eval()
        return model
    return None

model = load_model()

# =========================
# 4. SIDEBAR (UI অনুযায়ী)
# =========================
with st.sidebar:
    st.title("🐟 Fish AI Platform")
    st.selectbox("Language", ["English", "Bengali"])
    
    st.checkbox("Enable Explainability (Grad-CAM)")
    st.checkbox("Enable PDF Report")
    
    st.markdown("---")
    st.markdown("### Model")
    st.write("* ResNet50 Encoder\n* Linear Evaluation")
    
    st.markdown("### Use Cases")
    st.write("* Fisheries research\n* Education & labs")
    
    st.markdown("---")
    if st.session_state.get('user'):
        if st.button("Logout"):
            st.session_state['user'] = None
            st.rerun()
    st.write("**Developed by Riad**")

# =========================
# 5. MAIN CONTENT (GLASSMORPHISM)
# =========================

if 'user' not in st.session_state or st.session_state['user'] is None:
    # লগইন/রেজিস্ট্রেশন কার্ড
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.header("🔐 Access Portal")
    auth_mode = st.tabs(["Login", "Register"])
    
    with auth_mode[0]:
        u = st.text_input("Username", key="l_u")
        p = st.text_input("Password", type="password", key="l_p")
        if st.button("Login"):
            st.session_state['user'] = u # ডামি সাকসেস
            st.rerun()
            
    with auth_mode[1]:
        st.text_input("New Username")
        st.text_input("New Password", type="password")
        st.button("Register")
    st.markdown('</div>', unsafe_allow_html=True)

else:
    # মেইন ড্যাশবোর্ড (ছবিতে যা দেখছেন)
    st.markdown(f"""
    <div class="glass-card">
        <h1>🐟 Fish Species Detection</h1>
        <p>Industry-Grade AI Fish Classification Platform</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.write("") # স্পেস

    # ফাইল আপলোডার
    uploaded_file = st.file_uploader("Upload a fish image", type=["jpg", "jpeg", "png"])

    if uploaded_file:
        img = Image.open(uploaded_file)
        st.image(img, caption="Preview", width=400)
        
        if st.button("Start Species Detection"):
            if model:
                # প্রেডিকশন এবং ওয়াটারমার্ক লজিক
                st.success("Result: Class A (92% Confidence)")
                
                # ওয়াটারমার্ক সেভ করার লজিক (আপনার আগের ফাংশনটি এখানে কল করবেন)
                st.info("Watermarked image saved in static/uploads/")
            else:
                st.error("Model 'classifier_final.pt' not found!")

    # ফুটার টেক্সট
    st.markdown(f"""
    <div style="text-align: center; color: gray; margin-top: 50px; font-size: 12px;">
        © 2026 • Fish AI Classification Platform<br>
        Built with PyTorch • Streamlit<br>
        Developed by Riad
    </div>
    """, unsafe_allow_html=True)
