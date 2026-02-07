import streamlit as st
import os
import uuid
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
from werkzeug.security import generate_password_hash, check_password_hash

# ==========================================
# 1. INITIAL SETUP & PATHS
# ==========================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models", "classifier_final.pt")
UPLOAD_DIR = os.path.join(BASE_DIR, "static", "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)

st.set_page_config(page_title="Fish AI - Enterprise", page_icon="🐟", layout="wide")

# ==========================================
# 2. UI ENGINE (GLASSMORPHISM)
# ==========================================
def apply_theme():
    st.markdown(f"""
    <style>
    .stApp {{
        background: linear-gradient(rgba(0,0,0,0.8), rgba(0,0,0,0.8)), 
                    url("https://images.unsplash.com/photo-1524704654690-b56c05c78a00?q=80&w=2069");
        background-size: cover; background-attachment: fixed;
    }}
    .glass-card {{
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(15px);
        border-radius: 20px; border: 1px solid rgba(255, 255, 255, 0.1);
        padding: 30px; margin-bottom: 20px; color: white; text-align: center;
    }}
    div.stButton > button {{
        background: linear-gradient(90deg, #00C2FF, #0072FF);
        color: white; border: none; border-radius: 10px; height: 3.5em; width: 100%; font-weight: bold;
    }}
    [data-testid="stSidebar"] {{ background-color: #0e1117 !important; }}
    </style>
    """, unsafe_allow_html=True)

apply_theme()

# ==========================================
# 3. AI CORE (FIXES KEY MISMATCH)
# ==========================================
@st.cache_resource
def load_production_model():
    if not os.path.exists(MODEL_PATH):
        return None, "Model path error"
    try:
        # ResNet50 for 21 Classes
        model = models.resnet50(weights=None)
        model.fc = nn.Linear(model.fc.in_features, 21)
        
        state_dict = torch.load(MODEL_PATH, map_location=torch.device('cpu'))
        
        # SimCLR Key Mapping Fix
        new_state_dict = {k.replace("encoder.", ""): v for k, v in state_dict.items()}
        
        model.load_state_dict(new_state_dict, strict=False)
        model.eval()
        return model, "Operational"
    except Exception as e:
        return None, str(e)

model, model_status = load_production_model()

CLASS_NAMES = [
    "Baim", "Bata", "Batasio(tenra)", "Chitul", "Croaker(Poya)", 
    "Hilsha", "Kajoli", "Meni", "Pabda", "Poli", "Puti", 
    "Rita", "Rui", "Rupchada", "Silver Carp", "Telapiya", 
    "carp", "k", "kaikka", "koral", "shrimp"
]

# ==========================================
# 4. AUTHENTICATION SYSTEM
# ==========================================
if 'USERS' not in st.session_state:
    st.session_state['USERS'] = {"admin": generate_password_hash("admin123")} # Default Admin
if 'user' not in st.session_state:
    st.session_state['user'] = None

def login_user(u, p):
    if u in st.session_state['USERS'] and check_password_hash(st.session_state['USERS'][u], p):
        st.session_state['user'] = u
        return True
    return False

# ==========================================
# 5. NAVIGATION & PAGES
# ==========================================
with st.sidebar:
    st.title("🐟 Fish AI Platform")
    if st.session_state['user']:
        st.success(f"Logged in: {st.session_state['user']}")
        page = st.radio("Navigation", ["Dashboard", "Logout"])
    else:
        page = st.radio("Navigation", ["Login", "Register"])
    
    st.markdown("---")
    st.write("**Model:** ResNet50 SimCLR")
    st.write("**Status:**", model_status)
    st.write("Developed by **Riad**")

# --- AUTH PAGES ---
if page == "Register":
    st.markdown('<div class="glass-card"><h2>Create Account</h2></div>', unsafe_allow_html=True)
    new_u = st.text_input("Username")
    new_p = st.text_input("Password", type="password")
    if st.button("Sign Up"):
        st.session_state['USERS'][new_u] = generate_password_hash(new_p)
        st.success("Account created! Go to Login.")

elif page == "Login":
    st.markdown('<div class="glass-card"><h2>Secure Login</h2></div>', unsafe_allow_html=True)
    u = st.text_input("Username")
    p = st.text_input("Password", type="password")
    if st.button("Access Dashboard"):
        if login_user(u, p): st.rerun()
        else: st.error("Invalid credentials")

elif page == "Logout":
    st.session_state['user'] = None
    st.rerun()

# --- MAIN DASHBOARD ---
elif page == "Dashboard":
    st.markdown('<div class="glass-card"><h1>Fish Species Detection</h1><p>Professional Neural Analysis Portal</p></div>', unsafe_allow_html=True)
    
    if model is None:
        st.error(f"System Offline: Model not found at {MODEL_PATH}")
    
    uploaded_file = st.file_uploader("Upload image for analysis", type=["jpg", "png", "jpeg"])

    if uploaded_file:
        col1, col2 = st.columns(2)
        with col1:
            st.image(uploaded_file, caption="Input Stream", use_container_width=True)
        
        with col2:
            if st.button("RUN AI DIAGNOSTICS"):
                if model:
                    with st.spinner("Analyzing..."):
                        try:
                            # Image Pre-processing
                            transform = transforms.Compose([
                                transforms.Resize((224, 224)),
                                transforms.ToTensor(),
                                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                            ])
                            img = Image.open(uploaded_file).convert('RGB')
                            input_tensor = transform(img).unsqueeze(0)

                            # Prediction
                            with torch.no_grad():
                                output = model(input_tensor)
                                prob = torch.nn.functional.softmax(output[0], dim=0)
                                conf, idx = torch.max(prob, 0)

                            res = CLASS_NAMES[idx.item()]
                            
                            st.markdown(f'''
                                <div class="glass-card" style="border: 2px solid #00C2FF;">
                                    <h2 style="color: #00C2FF;">Species: {res}</h2>
                                    <h3>Confidence: {conf.item()*100:.2f}%</h3>
                                </div>
                            ''', unsafe_allow_html=True)
                        except Exception as e:
                            st.error(f"Prediction Error: {e}")

st.markdown('<div style="text-align:center; color:gray; margin-top:50px;">© 2026 Fish AI Classification Platform | Production Build</div>', unsafe_allow_html=True)
