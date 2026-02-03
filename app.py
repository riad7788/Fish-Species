import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

# --- ১. এনকোডার আর্কিটেকচার (ResNet50) ---
def get_encoder():
    # নোটবুক অনুযায়ী ResNet50 বেস মডেল
    encoder = models.resnet50(weights=None)
    encoder.fc = nn.Identity() 
    return encoder

# --- ২. আল্টিমেট মডেল লোডার (আপনার নতুন লিঙ্ক সহ) ---
@st.cache_resource
def load_full_model():
    device = torch.device("cpu")
    
    # আপনার Hugging Face লিঙ্কসমূহ
    ENCODER_URL = "https://huggingface.co/riad300/fish-simclr-encoder/resolve/main/encoder_simclr.pt"
    # আপনার নতুন এবং সঠিক ফাইল লিঙ্ক
    CLASSIFIER_URL = "https://huggingface.co/riad300/fish-simclr-encoder/resolve/main/classifier_final.pt"
    
    # এনকোডার লোড করা
    encoder = get_encoder()
    e_state = torch.hub.load_state_dict_from_url(ENCODER_URL, map_location=device, check_hash=False)
    encoder.load_state_dict(e_state)
    
    # ক্লাসিফায়ার লোড করা
    classifier = nn.Linear(2048, 21)
    c_state = torch.hub.load_state_dict_from_url(CLASSIFIER_URL, map_location=device, check_hash=False)
    
    # ওয়েটস ম্যাপিং (যাতে কোনো Key Missing এরর না আসে)
    new_state = {}
    for k, v in c_state.items():
        name = k.replace('fc.', '') # 'fc.weight' -> 'weight'
        new_state[name] = v
    
    classifier.load_state_dict(new_state)
    
    encoder.eval()
    classifier.eval()
    return encoder, classifier

# --- ৩. ইউজার ইন্টারফেস ---
st.set_page_config(page_title="Fish AI Expert", page_icon="🐟")
st.title("🐟 Fish Species AI Classifier")
st.markdown("২১টি প্রজাতির মাছ শনাক্ত করতে ছবি আপলোড করুন।")

try:
    encoder, classifier = load_full_model()
    st.sidebar.success("মডেল এখন ১০০% রেডি!")
except Exception as e:
    st.error(f"লোডিং এরর: {e}")

# আপনার নোটবুকের ২১টি মাছের নামের লিস্ট
CLASSES = [
    "Biam", "Bata", "Batasio(tenra)", "Chitul", "Croaker(Poya)", "Hilsha",
    "Kajoli", "Meni", "Pabda", "Poli", "Puti", "Rita", "Rui", "Rupchanda",
    "Silver Carp", "Telapiya", "carp", "Koi", "kaikka", "koral", "shrimp"
]

uploaded_file = st.file_uploader("একটি ছবি সিলেক্ট করুন", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)
    
    # ইমেজ প্রি-প্রসেসিং (ResNet50 স্ট্যান্ডার্ড)
    tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    with torch.no_grad():
        input_tensor = tf(image).unsqueeze(0)
        # ফিচার এক্সট্রাকশন
        features = encoder(input_tensor)
        # ক্লাসিফিকেশন
        outputs = classifier(features)
        probs = torch.softmax(outputs, dim=1)
        confidence, idx = torch.max(probs, 1)
    
    # ফাইনাল রেজাল্ট
    st.success(f"### শনাক্ত করা হয়েছে: **{CLASSES[idx.item()]}**")
    st.info(f"কনফিডেন্স লেভেল: **{confidence.item():.2% Prime}**")
