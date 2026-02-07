import streamlit as st
import os
import uuid
import logging
import torch
from PIL import Image, ImageDraw, ImageFont
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename

# =========================
# 1. CONFIG & PATHS
# =========================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, "static/uploads")
MODEL_FOLDER = os.path.join(BASE_DIR, "models")
# আপনার ওয়াটারমার্ক ছবির সঠিক পাথ
WATERMARK_PATH = os.path.join(BASE_DIR, "static", "watermark.png") 

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")

# =========================
# 2. LOAD MODEL (classifier_final.pt)
# =========================
# আপনার ফাইলের নাম অনুযায়ী এখানে আপডেট করা হয়েছে
MODEL_FILE_NAME = "classifier_final.pt" 
MODEL_PATH = os.path.join(MODEL_FOLDER, MODEL_FILE_NAME)

@st.cache_resource
def load_final_model():
    if not os.path.exists(MODEL_PATH):
        logging.error(f"Error: {MODEL_FILE_NAME} not found in {MODEL_FOLDER}")
        return None
    try:
        # মডেল লোড
        model = torch.load(MODEL_PATH, map_location="cpu")
        model.eval()
        logging.info(f"{MODEL_FILE_NAME} loaded successfully!")
        return model
    except Exception as e:
        logging.error(f"Failed to load model: {e}")
        return None

model = load_final_model()

# =========================
# 3. WATERMARK ENGINE
# =========================
def apply_advanced_watermark(base_image_path, output_path, result_text):
    # মেইন ছবি ওপেন করা
    base = Image.open(base_image_path).convert("RGBA")
    
    # নতুন একটি লেয়ার তৈরি করা ওয়াটারমার্কের জন্য
    txt_layer = Image.new("RGBA", base.size, (255, 255, 255, 0))
    draw = ImageDraw.Draw(txt_layer)
    
    # ১. ওয়াটারমার্ক ইমেজ (PNG) বসানো
    if os.path.exists(WATERMARK_PATH):
        wm_logo = Image.open(WATERMARK_PATH).convert("RGBA")
        # লোগো সাইজ অ্যাডজাস্ট করা (ছবির ২০%)
        wm_logo.thumbnail((base.width // 4, base.height // 4))
        # একদম মাঝখানে বা নিচে ডানে বসানো (নিচে ডানে সেট করলাম)
        position = (base.width - wm_logo.width - 20, base.height - wm_logo.height - 20)
        base.paste(wm_logo, position, wm_logo)
    
    # ২. রেজাল্ট টেক্সট লেখা
    try:
        # ভালো ফন্ট না থাকলে ডিফল্ট কাজ করবে
        draw.text((30, 30), f"PREDICTION: {result_text}", fill=(255, 255, 255, 180)) 
    except:
        draw.text((30, 30), f"PREDICTION: {result_text}", fill="white")

    # লেয়ার কম্বাইন করা
    out = Image.alpha_composite(base, txt_layer)
    out.convert("RGB").save(output_path, "JPEG")
    return output_path

# =========================
# 4. STREAMLIT UI
# =========================
if 'USERS' not in st.session_state: st.session_state['USERS'] = {}
if 'user' not in st.session_state: st.session_state['user'] = None

st.sidebar.title("Navigation")
if st.session_state['user']:
    choice = st.sidebar.selectbox("Menu", ["Dashboard", "Profile", "Logout"])
else:
    choice = st.sidebar.selectbox("Menu", ["Login", "Register"])

# --- AUTH LOGIC ---
if choice == "Register":
    st.title("Register")
    u = st.text_input("New Username")
    p = st.text_input("New Password", type="password")
    if st.button("Create Account"):
        st.session_state['USERS'][u] = {"password": generate_password_hash(p)}
        st.success("User Registered!")

elif choice == "Login":
    st.title("User Login")
    u = st.text_input("Username")
    p = st.text_input("Password", type="password")
    if st.button("Login"):
        user = st.session_state['USERS'].get(u)
        if user and check_password_hash(user["password"], p):
            st.session_state['user'] = u
            st.rerun()
        else:
            st.error("Wrong info!")

# --- DASHBOARD (Main logic) ---
elif choice == "Dashboard":
    st.title(f"Welcome {st.session_state['user']}")
    
    if model is None:
        st.warning(f"Warning: {MODEL_FILE_NAME} is missing. Using dummy mode.")

    uploaded_file = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])

    if uploaded_file:
        # ফাইল সেভ করা (UUID দিয়ে আপনার অরিজিনাল লজিক)
        ext = uploaded_file.name.split('.')[-1]
        unique_name = f"{uuid.uuid4()}.{ext}"
        temp_path = os.path.join(UPLOAD_FOLDER, f"raw_{unique_name}")
        final_path = os.path.join(UPLOAD_FOLDER, f"wm_{unique_name}")

        with open(temp_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        if st.button("Analyze & Add Watermark"):
            with st.spinner("Processing..."):
                # প্রেডিকশন
                # এখানে আপনার মডেলের ইনপুট প্রি-প্রসেসিং কোড বসাবেন
                pred_label = "Class A" # Dummy result
                conf = "95%"

                # ওয়াটারমার্ক অ্যাপ্লাই
                result_str = f"{pred_label} | {conf}"
                processed_img = apply_advanced_watermark(temp_path, final_path, result_str)

                # রেজাল্ট শো
                st.image(processed_img, caption="Final Result with Watermark")
                st.success(f"Successfully Analyzed by {st.session_state['user']}")
                
                # ফাইল ডাউনলোড বাটন (অপশনাল)
                with open(processed_img, "rb") as file:
                    st.download_button("Download Marked Image", file, file_name=f"result_{unique_name}")

elif choice == "Logout":
    st.session_state['user'] = None
    st.rerun()
