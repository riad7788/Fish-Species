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
# 1. CONFIGURATION & PATHS
# ==========================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models", "classifier_final.pt")
UPLOAD_FOLDER = os.path.join(BASE_DIR, "static", "uploads")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")

# Industry Standard Page Setup
st.set_page_config(
    page_title="Fish AI - Industry Grade Classification",
    page_icon="🐟",
    layout="wide"
)

# ==========================================
# 2. CUSTOM CSS (Glassmorphism & Professional UI)
# ==========================================
def apply_ui_style():
    st.markdown("""
    <style>
    /* Background setup */
    .stApp {
        background: linear-gradient(rgba(0,0,0,0.75), rgba(0,0,0,0.75)), 
                    url("https://images.unsplash.com/photo-1524704654690-b56c05c78a00?q=80&w=2069");
        background-size: cover;
        background-attachment: fixed;
    }
    
    /* Glassmorphism Card Style */
    .glass-card {
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(15px);
        -webkit-backdrop-filter: blur(15px);
        border-radius: 20px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        padding: 40px;
        margin-bottom: 25px;
        text-align: center;
        color: white;
    }

    /* Sidebar Dark Theme */
    [data-testid="stSidebar"] {
        background-color: #0e1117 !important;
        border-right: 1px solid rgba(255,255,255,0.1);
    }

    /* Custom Buttons */
    div.stButton > button {
        background: linear-gradient(90deg, #00C2FF, #0072FF);
        color: white;
        border: none;
        border-radius: 12px;
        padding: 12px 24px;
        font-weight: 600;
        transition: 0.3s;
        width: 100%;
    }
    div.stButton > button:hover {
        transform: scale(1.02);
        box-shadow: 0 4px 15px rgba(0, 194, 255, 0.4);
    }
    </style>
    """, unsafe_allow_html=True)

apply_ui_style()

# ==========================================
# 3. CORE AI ENGINE (Model Loader & Classes)
# ==========================================
@st.cache_resource
def load_production_model():
    if not os.path.exists(MODEL_PATH):
        return None, f"Model not found at: {MODEL_PATH}"
    try:
        # Load model to CPU for cloud stability
        model = torch.load(MODEL_PATH, map_location=torch.device('cpu'))
        model.eval()
        return model, "Ready"
    except Exception as e:
        return None, str(e)

model, status = load_production_model()

# আপনার দেওয়া সেই ফিশ ক্লাসের লিস্ট
CLASS_NAMES = [
    "Baim", "Bata", "Batasio(tenra)", "Chitul", "Croaker(Poya)", 
    "Hilsha", "Kajoli", "Meni", "Pabda", "Poli", "Puti", 
    "Rita", "Rui", "Rupchada", "Silver Carp", "Telapiya", 
    "carp", "k", "kaikka", "koral", "shrimp"
]

# ==========================================
# 4. AUTHENTICATION & SESSION
# ==========================================
if 'USERS' not in st.session_state: st.session_state['USERS'] = {}
if 'user' not in st.session_state: st.session_state['user'] = None

# ==========================================
# 5. SIDEBAR NAVIGATION
# ==========================================
with st.sidebar:
    st.markdown("## 🐟 Fish AI Platform")
    if st.session_state['user']:
        st.success(f"Active User: {st.session_state['user']}")
        nav = st.radio("Management", ["Dashboard", "Profile", "Logout"])
    else:
        nav = st.radio("Navigation", ["Home", "Login", "Register"])
    
    st.markdown("---")
    st.markdown("### Model Architecture")
    st.write("• ResNet50 Encoder\n• SimCLR V2 Training\n• PyTorch Framework")
    st.markdown("---")
    st.write("Developer: **Riad**")

# ==========================================
# 6. APP CONTROLLER (Pages)
# ==========================================

# --- Home Page ---
if nav == "Home":
    st.markdown('<div class="glass-card"><h1>Welcome to Fish AI</h1><p>The Future of Fisheries Research & Industry Detection</p></div>', unsafe_allow_html=True)

# --- Register ---
elif nav == "Register":
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.subheader("Account Registration")
    reg_u = st.text_input("Choose Username")
    reg_p = st.text_input("Set Password", type="password")
    if st.button("Create Account"):
        if reg_u and reg_p:
            st.session_state['USERS'][reg_u] = {"password": generate_password_hash(reg_p)}
            st.success("Account created! Please Login.")
    st.markdown('</div>', unsafe_allow_html=True)

# --- Login ---
elif nav == "Login":
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.subheader("Login to Dashboard")
    u = st.text_input("Username")
    p = st.text_input("Password", type="password")
    if st.button("Login"):
        user_data = st.session_state['USERS'].get(u)
        if user_data and check_password_hash(user_data["password"], p):
            st.session_state['user'] = u
            st.rerun()
        else: st.error("Incorrect Credentials!")
    st.markdown('</div>', unsafe_allow_html=True)

# --- Logout ---
elif nav == "Logout":
    st.session_state['user'] = None
    st.rerun()

# --- Dashboard (Main Application) ---
elif nav == "Dashboard":
    st.markdown('<div class="glass-card"><h1>🐟 Fish Species Detection</h1><p>Real-time AI Analysis Portal</p></div>', unsafe_allow_html=True)
    
    if model is None:
        st.error(f"❌ System Error: Model missing at {MODEL_PATH}")
    
    uploaded_file = st.file_uploader("", type=["jpg", "png", "jpeg"])

    if uploaded_file:
        # Secure file saving with UUID
        unique_name = f"{uuid.uuid4()}_{secure_filename(uploaded_file.name)}"
        f_path = os.path.join(UPLOAD_FOLDER, unique_name)
        with open(f_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        col1, col2 = st.columns([1, 1])
        with col1:
            st.image(uploaded_file, caption="Input Stream", use_container_width=True)
        
        with col2:
            if st.button("Run AI Analysis"):
                if model:
                    with st.spinner("Analyzing Morphology..."):
                        try:
                            # Pre-processing pipeline
                            preprocess = transforms.Compose([
                                transforms.Resize((224, 224)),
                                transforms.ToTensor(),
                                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                            ])
                            
                            img = Image.open(uploaded_file).convert('RGB')
                            input_tensor = preprocess(img).unsqueeze(0)

                            # Prediction Engine
                            with torch.no_grad():
                                output = model(input_tensor)
                                # Check output shape
                                if len(output.shape) > 1:
                                    prob = torch.nn.functional.softmax(output, dim=1)
                                    conf, pred_idx = torch.max(prob, 1)
                                else:
                                    prob = torch.nn.functional.softmax(output, dim=0)
                                    conf, pred_idx = torch.max(prob, 0)

                            fish_result = CLASS_NAMES[pred_idx.item()] if pred_idx.item() < len(CLASS_NAMES) else "Unknown"

                            # Display Results
                            st.markdown(f'''
                            <div class="glass-card" style="border: 2px solid #00C2FF; background: rgba(0, 194, 255, 0.1);">
                                <h2 style="color: #00C2FF; margin-bottom: 0px;">Species: {fish_result}</h2>
                                <p style="font-size: 20px;">Confidence Score: {conf.item()*100:.2f}%</p>
                            </div>
                            ''', unsafe_allow_html=True)
                            
                            logging.info(f"Analysis Success: {fish_result} by {st.session_state['user']}")

                        except Exception as e:
                            st.error(f"AI Engine Error: {str(e)}")
                else:
                    st.error("System Offline: Model not initialized.")

# --- Footer ---
st.markdown("""
<div style="text-align: center; color: rgba(255,255,255,0.4); font-size: 12px; margin-top: 60px; padding: 20px;">
    © 2026 • Fish AI Platform • Industry Grade Detection<br>
    Built with PyTorch, SIMCLR & Streamlit • Developed by Riad
</div>
""", unsafe_allow_html=True)
