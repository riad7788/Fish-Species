import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

# --- ১. সঠিক এনকোডার আর্কিটেকচার (resnet50) ---
def get_encoder():
    # সরাসরি ResNet50 বেস মডেল
    encoder = models.resnet50(weights=None)
    # আপনার নোটবুক অনুযায়ী fc লেয়ারকে Identity করা হয়েছে যাতে ২০৪৮ টি ফিচার পাওয়া যায়
    encoder.fc = nn.Identity() 
    return encoder

# --- ২. ক্লাসিফায়ার লোডিং লজিক (এরর ফিক্সড) ---
@st.cache_resource
def load_full_model():
    device = torch.device("cpu")
    
    # Hugging Face লিঙ্কসমূহ
    ENCODER_URL = "https://huggingface.co/riad300/fish-simclr-encoder/resolve/main/encoder_simclr.pt"
    CLASSIFIER_URL = "https://huggingface.co/riad300/fish-simclr-encoder/resolve/main/classifier.pt"
    
    # এনকোডার লোড
    encoder = get_encoder()
    e_state = torch.hub.load_state_dict_from_url(ENCODER_URL, map_location=device, check_hash=False)
    encoder.load_state_dict(e_state)
    
    # আপনার এরর অনুযায়ী ক্লাসিফায়ার সরাসরি একটি Linear লেয়ার
    classifier = nn.Linear(2048, 21)
    c_state = torch.hub.load_state_dict_from_url(CLASSIFIER_URL, map_location=device, check_hash=False)
    
    # এরর এড়াতে 'strict=False' এবং সরাসরি ওয়েটস ম্যাপিং
    try:
        classifier.load_state_dict(c_state)
    except:
        # যদি ফাইলটি শুধু ওয়েটস হয় (আপনার এরর অনুযায়ী এটিই সমস্যা)
        classifier.weight.data.copy_(c_state['weight'] if 'weight' in c_state else c_state)
        classifier.bias.data.copy_(c_state['bias'] if 'bias' in c_state else c_state)
        
    encoder.eval()
    classifier.eval()
    return encoder, classifier

# --- ৩. মেইন অ্যাপ ইন্টারফেস ---
st.set_page_config(page_title="Fish AI", page_icon="🐟")
st.title("🐟 Fish Species AI Classifier")

try:
    with st.spinner('মডেল লোড হচ্ছে...'):
        encoder, classifier = load_full_model()
    st.sidebar.success("মডেল রেডি!")
except Exception as e:
    st.error(f"Error: {e}")

# আপনার নোটবুকের ২১টি ক্লাসের নাম
CLASSES = [
    "Biam", "Bata", "Batasio(tenra)", "Chitul", "Croaker(Poya)", "Hilsha",
    "Kajoli", "Meni", "Pabda", "Poli", "Puti", "Rita", "Rui", "Rupchanda",
    "Silver Carp", "Telapiya", "carp", "Koi", "kaikka", "koral", "shrimp"
]

uploaded_file = st.file_uploader("একটি মাছের ছবি আপলোড করুন", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)
    
    # ইমেজ ট্রান্সফর্মেশন
    tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    with torch.no_grad():
        input_tensor = tf(image).unsqueeze(0)
        # এনকোডার থেকে ফিচার বের করা
        features = encoder(input_tensor)
        # ক্লাসিফায়ার থেকে প্রেডিকশন
        outputs = classifier(features)
        probs = torch.softmax(outputs, dim=1)
        confidence, idx = torch.max(probs, 1)
    
    st.success(f"### রেজাল্ট: {CLASSES[idx.item()]}")
    st.info(f"কনফিডেন্স: {confidence.item():.2%}")
