import streamlit as st
import os
import uuid
import logging
import torch
import torchvision.transforms as transforms
from PIL import Image
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename

# ==========================================
# 1. SMART PATH DETECTION (FIXED)
# ==========================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# আপনার গিটহাবের ফোল্ডার স্ট্রাকচার অনুযায়ী পাথ সেট করা হচ্ছে
# এখানে আমরা models ফোল্ডারটি ছোট হাতের অক্ষরেই খুঁজবো
MODEL_PATH = os.path.join(BASE_DIR, "models", "classifier_final.pt")
UPLOAD_FOLDER = os.path.join(BASE_DIR, "static", "uploads")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")

st.set_page_config(page_title="Fish AI - Production", page_icon="🐟", layout="wide")

# ==========================================
# 2. PRO UI (GLASSMORPHISM)
# ==========================================
st.markdown("""
    <style>
    .stApp {
        background: linear-gradient(rgba(0,0,0,0.8), rgba(0,0,0,0.8)), 
                    url("https://images.unsplash.com/photo-1524704654690-b56c05c78a00?q=80&w=2069");
        background-size: cover; background-attachment: fixed;
    }
    .glass-card {
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(15px);
        border-radius: 20px; border: 1px solid rgba(255, 255, 255, 0.1);
        padding: 40px; text-align: center; color: white; margin-bottom: 20px;
    }
    [data-testid="stSidebar"] { background-color: #0e1117 !important; }
    div.stButton > button {
        background: linear-gradient(90deg, #00C2FF, #0072FF);
        color: white; border: none; border-radius: 10px; width: 100%; font-weight: bold;
    }
    </style>
""", unsafe_allow_html=True)

# ==========================================
# 3. MODEL LOADER
# ==========================================
@st.cache_resource
def load_production_model():
    # যদি পাথ না পায় তবে এরর হ্যান্ডেল করবে
    if not os.path.exists(MODEL_PATH):
        return None, f"Model file not found at: {MODEL_PATH}"
    try:
        # CPU তে লোড করা হচ্ছে যেন ক্লাউডে ক্র্যাশ না করে
        model = torch.load(MODEL_PATH, map_location=torch.device('cpu'))
        if hasattr(model, 'eval'):
            model.eval()
        return model, "Ready"
    except Exception as e:
        return None, str(e)

model, status_msg = load_production_model()

# আপনার দেওয়া সেই ২১টি মাছের ক্লাসের লিস্ট
CLASS_NAMES = [
    "Baim", "Bata", "Batasio(tenra)", "Chitul", "Croaker(Poya)", 
    "Hilsha", "Kajoli", "Meni", "Pabda", "Poli", "Puti", 
    "Rita", "Rui", "Rupchada", "Silver Carp", "Telapiya", 
    "carp", "k", "kaikka", "koral", "shrimp"
]

# ==========================================
# 4. AUTH & NAV
# ==========================================
if 'user' not in st.session_state: st.session_state['user'] = None

with st.sidebar:
    st.markdown("## 🐟 Fish AI Platform")
    if st.session_state['user']:
        st.success(f"User: {st.session_state['user']}")
        nav = st.radio("Go to", ["Dashboard", "Logout"])
    else:
        nav = st.radio("Go to", ["Login", "Register"])
    st.markdown("---")
    st.write("Developer: **Riad**")

# ==========================================
# 5. DASHBOARD & PREDICTION
# ==========================================
if nav == "Dashboard":
    st.markdown('<div class="glass-card"><h1>🐟 Fish Species Detection</h1></div>', unsafe_allow_html=True)
    
    # মডেল না পেলে এরর দেখাবে
    if model is None:
        st.error(f"System Offline: {status_msg}")
    
    uploaded_file = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])

    if uploaded_file:
        # UUID সেভিং
        u_id = f"{uuid.uuid4()}_{secure_filename(uploaded_file.name)}"
        path = os.path.join(UPLOAD_FOLDER, u_id)
        with open(path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        st.image(uploaded_file, width=400)

        if st.button("Run AI Analysis"):
            if model:
                with st.spinner("Processing..."):
                    try:
                        # প্রোসেসিং
                        preprocess = transforms.Compose([
                            transforms.Resize((224, 224)),
                            transforms.ToTensor(),
                            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                        ])
                        img = Image.open(uploaded_file).convert('RGB')
                        tensor = preprocess(img).unsqueeze(0)

                        # প্রেডিকশন
                        with torch.no_grad():
                            output = model(tensor)
                            prob = torch.nn.functional.softmax(output, dim=1)
                            conf, idx = torch.max(prob, 1)

                        res_name = CLASS_NAMES[idx.item()]
                        
                        st.markdown(f'''
                            <div class="glass-card" style="border: 2px solid #00C2FF;">
                                <h2 style="color: #00C2FF;">Species: {res_name}</h2>
                                <p>Confidence: {conf.item()*100:.2f}%</p>
                            </div>
                        ''', unsafe_allow_html=True)
                    except Exception as e:
                        st.error(f"Prediction Error: {e}")

# --- AUTH LOGIC (সংক্ষিপ্ত) ---
elif nav == "Login":
    st.markdown('<div class="glass-card"><h3>Login</h3>', unsafe_allow_html=True)
    u = st.text_input("User")
    if st.button("Enter"):
        st.session_state['user'] = u
        st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)

elif nav == "Logout":
    st.session_state['user'] = None
    st.rerun()
