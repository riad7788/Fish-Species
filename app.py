import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

# --- ১. এনকোডার আর্কিটেকচার (আপনার নোটবুক অনুযায়ী) ---
def get_encoder():
    # সরাসরি ResNet50 বেস মডেল
    encoder = models.resnet50(weights=None)
    encoder.fc = nn.Identity() # নোটবুক অনুযায়ী fc বাদ দেওয়া হয়েছে
    return encoder

# --- ২. মডেল লোডিং (Hugging Face থেকে) ---
@st.cache_resource
def load_full_model():
    device = torch.device("cpu")
    
    # এনকোডার এবং নতুন ক্লাসিফায়ার লিঙ্ক
    ENCODER_URL = "https://huggingface.co/riad300/fish-simclr-encoder/resolve/main/encoder_simclr.pt"
    CLASSIFIER_URL = "https://huggingface.co/riad300/fish-simclr-encoder/resolve/main/classifier_final.pt"
    
    # এনকোডার লোড
    encoder = get_encoder()
    e_state = torch.hub.load_state_dict_from_url(ENCODER_URL, map_location=device, check_hash=False)
    encoder.load_state_dict(e_state)
    
    # ক্লাসিফায়ার লোড (আপনার এরর হ্যান্ডেল করার জন্য)
    classifier = nn.Linear(2048, 21)
    c_state = torch.hub.load_state_dict_from_url(CLASSIFIER_URL, map_location=device, check_hash=False)
    
    # নামের অমিল থাকলে তা ঠিক করা (যেমন: fc.weight কে weight বানানো)
    new_state = {}
    for k, v in c_state.items():
        name = k.replace('fc.', '') 
        new_state[name] = v
    
    classifier.load_state_dict(new_state)
    
    encoder.eval()
    classifier.eval()
    return encoder, classifier

# --- ৩. মেইন ইউজার ইন্টারফেস ---
st.set_page_config(page_title="Fish AI Classifier", layout="centered")
st.title("🐟 Fish Species AI Expert")

# মডেল লোড করা
try:
    encoder, classifier = load_full_model()
    st.sidebar.success("মডেল এখন ১০০% রেডি!")
except Exception as e:
    st.error(f"মডেল লোডিং এরর: {e}")

# আপনার নোটবুকের ২১টি মাছের নামের লিস্ট
CLASSES = [
    "Biam", "Bata", "Batasio(tenra)", "Chitul", "Croaker(Poya)", "Hilsha",
    "Kajoli", "Meni", "Pabda", "Poli", "Puti", "Rita", "Rui", "Rupchanda",
    "Silver Carp", "Telapiya", "carp", "Koi", "kaikka", "koral", "shrimp"
]

file = st.file_uploader("মাছের ছবি আপলোড করুন", type=["jpg", "jpeg", "png"])

if file:
    img = Image.open(file).convert("RGB")
    st.image(img, use_container_width=True)
    
    # প্রি-প্রসেসিং (ResNet50 এর স্ট্যান্ডার্ড মান)
    tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    with torch.no_grad():
        input_data = tf(img).unsqueeze(0)
        # ফিচার এক্সট্রাকশন
        feats = encoder(input_data)
        # প্রেডিকশন
        output = classifier(feats)
        prob, idx = torch.max(torch.softmax(output, dim=1), 1)
    
    # সুন্দর করে আউটপুট দেখানো
    st.success(f"### শনাক্ত করা হয়েছে: **{CLASSES[idx.item()]}**")
    st.info(f"কনফিডেন্স লেভেল: **{prob.item()*100:.2f}%**")
