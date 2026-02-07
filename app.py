import streamlit as st
import os
import uuid
import logging
import torch
from PIL import Image, ImageDraw, ImageFont
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename

# =========================
# 1. BASIC CONFIG
# =========================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, "static/uploads")
MODEL_FOLDER = os.path.join(BASE_DIR, "models")
# আপনার ওয়াটারমার্ক ইমেজের পাথ (নিশ্চিত করুন এই ফাইলটি আছে)
WATERMARK_PATH = os.path.join(BASE_DIR, "static/watermark.png") 

ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg"}
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

# =========================
# 2. SESSION STATE
# =========================
if 'USERS' not in st.session_state: st.session_state['USERS'] = {}
if 'user' not in st.session_state: st.session_state['user'] = None

# =========================
# 3. LOAD MODEL
# =========================
MODEL_PATH = os.path.join(MODEL_FOLDER, "classifier.pt")
CLASS_NAMES = ["Class A", "Class B", "Class C"]

@st.cache_resource
def load_model():
    try:
        if os.path.exists(MODEL_PATH):
            model = torch.load(MODEL_PATH, map_location="cpu")
            model.eval()
            logging.info("Model loaded successfully")
            return model
    except Exception as e:
        logging.error(f"Model loading failed: {e}")
    return None

model = load_model()

# =========================
# 4. WATERMARK FUNCTION (ইমেজের ওপর লেখা ও লোগো বসানো)
# =========================
def apply_watermark(input_image_path, output_image_path, text_result):
    img = Image.open(input_image_path).convert("RGBA")
    draw = ImageDraw.Draw(img)
    
    # ১. ওয়াটারমার্ক ইমেজ বসানো (যদি থাকে)
    if os.path.exists(WATERMARK_PATH):
        mark = Image.open(WATERMARK_PATH).convert("RGBA")
        # ওয়াটারমার্ক রিসাইজ করা (ছবির ১০%)
        mark.thumbnail((img.width // 5, img.height // 5))
        img.paste(mark, (img.width - mark.width - 10, img.height - mark.height - 10), mark)
    
    # ২. ছবির ওপর টেক্সট লেখা (Result)
    try:
        # ফন্ট লোড করার চেষ্টা (না থাকলে ডিফল্ট)
        font = ImageFont.load_default() 
        draw.text((20, 20), f"Result: {text_result}", fill=(255, 255, 255, 255), font=font)
    except:
        draw.text((20, 20), f"Result: {text_result}", fill="white")

    img.convert("RGB").save(output_image_path)
    return output_image_path

# =========================
# 5. UI & ROUTES
# =========================
st.sidebar.title("Main Menu")
if st.session_state['user']:
    page = st.sidebar.radio("Navigate", ["Dashboard", "Profile", "Logout"])
else:
    page = st.sidebar.radio("Navigate", ["Login", "Register"])

# --- REGISTER/LOGIN (আগের মতই) ---
if page == "Register":
    st.title("Register")
    u = st.text_input("Username")
    p = st.text_input("Password", type="password")
    if st.button("Sign Up"):
        st.session_state['USERS'][u] = {"password": generate_password_hash(p)}
        st.success("Done!")

elif page == "Login":
    st.title("Login")
    u = st.text_input("Username")
    p = st.text_input("Password", type="password")
    if st.button("Login"):
        user_data = st.session_state['USERS'].get(u)
        if user_data and check_password_hash(user_data["password"], p):
            st.session_state['user'] = u
            st.rerun()

# --- DASHBOARD (প্রেডিকশন ও ওয়াটারমার্ক) ---
elif page == "Dashboard":
    st.title(f"Welcome, {st.session_state['user']}")
    uploaded_file = st.file_uploader("Upload Image", type=["jpg","png","jpeg"])

    if uploaded_file:
        # ১. অরিজিনাল ফাইল সেভ
        filename = secure_filename(uploaded_file.name)
        unique_id = str(uuid.uuid4())
        temp_path = os.path.join(UPLOAD_FOLDER, f"temp_{unique_id}_{filename}")
        final_path = os.path.join(UPLOAD_FOLDER, f"processed_{unique_id}_{filename}")

        with open(temp_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        if st.button("Predict & Process"):
            if model is None:
                st.error("Model missing in /models/classifier.pt")
            else:
                # ২. প্রেডিকশন (আপনার অরিজিনাল লজিক)
                # এখানে আপনার preprocessing code বসাতে পারেন
                predicted_class = CLASS_NAMES[0] 
                confidence = 0.92

                # ৩. ওয়াটারমার্ক অ্যাপ্লাই করা
                result_text = f"{predicted_class} ({confidence*100}%)"
                processed_img_path = apply_watermark(temp_path, final_path, result_text)

                # ৪. ডিসপ্লে
                st.image(processed_img_path, caption="Processed Image with Watermark")
                st.success(f"Prediction: {predicted_class}")
                st.info(f"Confidence: {confidence}")
                
                # ৫. লগিং
                logging.info(f"User {st.session_state['user']} processed {filename}")

elif page == "Logout":
    st.session_state['user'] = None
    st.rerun()
