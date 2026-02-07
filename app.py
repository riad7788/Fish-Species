import streamlit as st
import os
import uuid
import logging
import torch
from functools import wraps
from PIL import Image
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename

# =========================
# 1. BASIC CONFIG (আপনার কোড অনুযায়ী)
# =========================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, "static/uploads")
MODEL_FOLDER = os.path.join(BASE_DIR, "models")
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg"}

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# =========================
# 2. LOGGING (আপনার কোড অনুযায়ী)
# =========================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

# =========================
# 3. SESSION & DATABASE (USERS)
# =========================
# Streamlit-এ session_state ই Flask-এর session এবং USERS ডিকশনারির কাজ করবে
if 'USERS' not in st.session_state:
    st.session_state['USERS'] = {}
if 'user' not in st.session_state:
    st.session_state['user'] = None

# =========================
# 4. LOAD MODEL (আপনার কোড অনুযায়ী)
# =========================
MODEL_PATH = os.path.join(MODEL_FOLDER, "classifier.pt")
CLASS_NAMES = ["Class A", "Class B", "Class C"]

@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        logging.error("Model path does not exist")
        return None
    try:
        model = torch.load(MODEL_PATH, map_location="cpu")
        model.eval()
        logging.info("Model loaded successfully")
        return model
    except Exception as e:
        logging.error(f"Model loading failed: {e}")
        return None

model = load_model()

# =========================
# 5. HELPERS (আপনার কোড অনুযায়ী)
# =========================
def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

# =========================
# 6. SIDEBAR NAVIGATION (ROUTES)
# =========================
st.sidebar.title("App Navigation")
if st.session_state['user']:
    page = st.sidebar.radio("Go to", ["Dashboard", "Profile", "Logout"])
else:
    page = st.sidebar.radio("Go to", ["Home", "Login", "Register"])

# =========================
# 7. ROUTES LOGIC (ALL FUNCTIONS)
# =========================

# ---------- HOME ----------
if page == "Home":
    st.title("Welcome to Classifier")
    st.write("Please login or register to continue.")

# ---------- REGISTER ----------
elif page == "Register":
    st.title("Register Account")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    if st.button("Register"):
        if username in st.session_state['USERS']:
            st.error("User already exists")
        elif username and password:
            st.session_state['USERS'][username] = {
                "password": generate_password_hash(password)
            }
            st.success("Registration successful. Please login.")
            logging.info(f"New user registered: {username}")
        else:
            st.warning("All fields are required.")

# ---------- LOGIN ----------
elif page == "Login":
    st.title("Login")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    if st.button("Login"):
        user = st.session_state['USERS'].get(username)
        if not user or not check_password_hash(user["password"], password):
            st.error("Invalid credentials")
        else:
            st.session_state['user'] = username
            st.success("Login successful")
            logging.info(f"User logged in: {username}")
            st.rerun()

# ---------- LOGOUT ----------
elif page == "Logout":
    st.session_state['user'] = None
    st.info("Logged out successfully")
    st.rerun()

# ---------- PROFILE ----------
elif page == "Profile":
    st.title("User Profile")
    st.write(f"Logged in as: **{st.session_state['user']}**")

# ---------- DASHBOARD & PREDICTION (MAIN FEATURE) ----------
elif page == "Dashboard":
    st.title(f"Dashboard - {st.session_state['user']}")
    
    # Predict Logic
    st.subheader("Upload Image for Prediction")
    file = st.file_uploader("Choose an image...", type=["png", "jpg", "jpeg"])

    if file:
        if not allowed_file(file.name):
            st.error("Invalid file type")
        else:
            # UUID এবং Secure Filename দিয়ে ফাইল সেভ
            filename = secure_filename(file.name)
            unique_name = f"{uuid.uuid4()}_{filename}"
            filepath = os.path.join(UPLOAD_FOLDER, unique_name)
            
            # ফাইল রাইট করা
            with open(filepath, "wb") as f:
                f.write(file.getbuffer())

            # Prediction UI (আপনার result.html এর সব ফিচার)
            if st.button("Run Prediction"):
                if model is None:
                    st.error("Model not available")
                else:
                    with st.spinner('Analyzing...'):
                        # -------- MODEL INFERENCE (DUMMY logic from your code) --------
                        predicted_class = CLASS_NAMES[0]
                        confidence = 0.92

                        # Result Display (Watermark style display)
                        st.success("Analysis Complete!")
                        
                        col1, col2 = st.columns([1, 1])
                        with col1:
                            st.image(file, caption="Uploaded Image", use_container_width=True)
                        
                        with col2:
                            st.markdown("### Result Details")
                            st.info(f"**Prediction:** {predicted_class}")
                            st.metric("Confidence", f"{confidence*100}%")
                            st.caption(f"File saved as: {unique_name}")
                            
                            # Watermark Style Footer
                            st.markdown("---")
                            st.markdown(f"<p style='opacity: 0.5; text-align: center;'>Generated by AI Classifier - {st.session_state['user']}</p>", unsafe_allow_state_allowed=True)
