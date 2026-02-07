import streamlit as st
import os
import uuid
import logging
import torch
from PIL import Image, ImageDraw, ImageFont
from werkzeug.security import generate_password_hash, check_password_hash

# =========================
# 1. CONFIG & LOGGING
# =========================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, "static/uploads")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")

# Page Config
st.set_page_config(page_title="Fish AI Platform", layout="wide")

# =========================
# 2. FIXED CUSTOM CSS (হুবহু ছবির মতো ডিজাইন)
# =========================
def local_css():
    st.markdown("""
    <style>
    /* ব্যাকগ্রাউন্ড সেটআপ */
    .stApp {
        background: linear-gradient(rgba(0,0,0,0.6), rgba(0,0,0,0.6)), 
                    url("https://images.unsplash.com/photo-1524704654690-b56c05c78a00?q=80&w=2069");
        background-size: cover;
        background-position: center;
        background-attachment: fixed;
    }
    
    /* গ্লাস কার্ড স্টাইল */
    .glass-card {
        background: rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(15px);
        -webkit-backdrop-filter: blur(15px);
        border-radius: 20px;
        border: 1px solid rgba(255, 255, 255, 0.2);
        padding: 50px;
        margin: 20px auto;
        text-align: center;
        color: white;
        max-width: 800px;
    }

    /* সাইডবার ডার্ক লুক */
    [data-testid="stSidebar"] {
        background-color: #161a24;
    }

    /* বাটন ডিজাইন */
    div.stButton > button {
        background: #00a0ff;
        color: white;
        border-radius: 8px;
        border: none;
        width: 100%;
        font-weight: bold;
    }
    </style>
    """, unsafe_allow_html=True) # এখানে ভুল ছিল, এখন ঠিক করা হয়েছে

local_css()

# =========================
# 3. LOAD MODEL (classifier_final.pt)
# =========================
MODEL_PATH = os.path.join(BASE_DIR, "models", "classifier_final.pt")

@st.cache_resource
def load_my_model():
    if os.path.exists(MODEL_PATH):
        try:
            model = torch.load(MODEL_PATH, map_location="cpu")
            model.eval()
            return model
        except Exception as e:
            logging.error(f"Model Error: {e}")
    return None

model = load_my_model()

# =========================
# 4. SIDEBAR (ছবির মতো মেনু)
# =========================
with st.sidebar:
    st.markdown("### 🐟 Fish AI Platform")
    st.selectbox("Language", ["English", "Bengali"])
    
    st.checkbox("Enable Explainability (Grad-CAM)")
    st.checkbox("Enable PDF Report")
    
    st.markdown("---")
    st.markdown("**Model Details**")
    st.write("• ResNet50 Encoder\n• Linear Evaluation")
    
    st.markdown("---")
    st.markdown("**Developed by Riad**")

# =========================
# 5. MAIN CONTENT
# =========================

if 'user' not in st.session_state:
    st.session_state['user'] = None

if st.session_state['user'] is None:
    # লগইন বক্স (গ্লাস ইফেক্ট)
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.header("🔑 Login to Platform")
    user_in = st.text_input("Username")
    pass_in = st.text_input("Password", type="password")
    if st.button("Enter Platform"):
        st.session_state['user'] = user_in
        st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)

else:
    # মেইন ড্যাশবোর্ড (ছবির মতো লুক)
    st.markdown(f"""
    <div class="glass-card">
        <h1>🐟 Fish Species Detection</h1>
        <p>Industry-Grade AI Fish Classification Platform</p>
        <hr style="border: 0.5px solid rgba(255,255,255,0.2)">
    </div>
    """, unsafe_allow_html=True)

    # ফাইল আপলোড অংশ
    uploaded_file = st.file_uploader("", type=["jpg", "png", "jpeg"])

    if uploaded_file:
        # UUID দিয়ে ফাইল সেভ লজিক
        ext = uploaded_file.name.split('.')[-1]
        unique_name = f"{uuid.uuid4()}.{ext}"
        save_path = os.path.join(UPLOAD_FOLDER, unique_name)
        
        with open(save_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        st.image(uploaded_file, width=300)

        if st.button("Browse files & Predict"):
            with st.spinner("Analyzing Fish Species..."):
                # আপনার মডেল থেকে রেজাল্ট আসবে এখানে
                res_class = "Salmon" # Dummy
                conf = "98.5%"
                
                st.markdown(f"""
                <div class="glass-card" style="padding: 20px;">
                    <h3>Result: {res_class}</h3>
                    <p>Confidence: {conf}</p>
                </div>
                """, unsafe_allow_html=True)
                logging.info(f"User {st.session_state['user']} predicted: {res_class}")

    # ফুটার (ছবির নিচের টেক্সট)
    st.markdown("""
    <div style="text-align: center; color: rgba(255,255,255,0.5); font-size: 13px; margin-top: 30px;">
        © 2026 • Fish AI Classification Platform<br>
        Built with PyTorch • SimCLR • Streamlit<br>
        Developed by Riad
    </div>
    """, unsafe_allow_html=True)
