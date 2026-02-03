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

# --- ২. মডেল লোডার (আপনার নতুন fish_expert_weights.pt সহ) ---
@st.cache_resource
def load_bd_expert_model():
    device = torch.device("cpu")
    
    # আপনার Hugging Face লিঙ্কসমূহ
    ENCODER_URL = "https://huggingface.co/riad300/fish-simclr-encoder/resolve/main/encoder_simclr.pt"
    # আপনার নতুন আপলোড করা সঠিক ফাইল লিঙ্ক
    CLASSIFIER_URL = "https://huggingface.co/riad300/fish-simclr-encoder/resolve/main/fish_expert_weights.pt"
    
    # এনকোডার লোড করা
    encoder = get_encoder()
    e_state = torch.hub.load_state_dict_from_url(ENCODER_URL, map_location=device, check_hash=False)
    encoder.load_state_dict(e_state)
    
    # ক্লাসিফায়ার লোড করা (২১টি মাছের জন্য)
    classifier = nn.Linear(2048, 21)
    c_state = torch.hub.load_state_dict_from_url(CLASSIFIER_URL, map_location=device, check_hash=False)
    
    # ওয়েটস ম্যাপিং (যাতে Missing Key এরর না আসে)
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
st.title("🐟 দেশি মাছ শনাক্তকারী (Expert Mode)")
st.markdown("আপনার ট্রেনিং করা ২১টি প্রজাতির মাছ শনাক্ত করতে ছবি আপলোড করুন।")

try:
    encoder, classifier = load_bd_expert_model()
    st.sidebar.success("মডেল এখন ১০০% রেডি!")
except Exception as e:
    st.error(f"লোডিং এরর: {e}")

# আপনার নোটবুকের ২১টি মাছের নামের লিস্ট
CLASSES = [
    "Baim (বাইন)", "Bata (বাটা)", "Batasio/Tengra (টেংরা)", "Chitul (চিতল)", 
    "Croaker/Poya (পোয়া)", "Hilsha (ইলিশ)", "Kajoli (কাজলী)", "Meni (মেনি)", 
    "Pabda (পাবদা)", "Poli (ফলি)", "Puti (পুঁটি)", "Rita (রিটা)", 
    "Rui (রুই)", "Rupchanda (রূপচাঁদা)", "Silver Carp (সিলভার কার্প)", 
    "Telapiya (তেলাপিয়া)", "Carp (কার্প)", "Koi (কৈ)", 
    "Kaikka (কাইক্কা)", "Koral (কোরাল)", "Shrimp (চিংড়ি)"
]

uploaded_file = st.file_uploader("একটি মাছের ছবি আপলোড করুন", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="বিশ্লেষণ করা হচ্ছে...", use_container_width=True)
    
    # ইমেজ প্রি-প্রসেসিং
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
    st.info(f"নিশ্চয়তা (Confidence): **{confidence.item()*100:.2f}%**")
