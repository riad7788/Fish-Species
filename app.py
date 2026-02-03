import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

# --- ১. মডেল আর্কিটেকচার (আপনার নোটবুক অনুযায়ী) ---
def get_encoder():
    # ResNet50 বেস যা ফিচার এক্সট্রাক্ট করবে
    encoder = models.resnet50(weights=None)
    encoder.fc = nn.Identity() 
    return encoder

# --- ২. মডেল লোডার (নতুন ফাইল লিঙ্ক সহ) ---
@st.cache_resource(show_spinner="নতুন মডেল লোড হচ্ছে, দয়া করে অপেক্ষা করুন...")
def load_expert_model():
    device = torch.device("cpu")
    
    # আপনার Hugging Face এর সঠিক লিঙ্কসমূহ
    ENCODER_URL = "https://huggingface.co/riad300/fish-simclr-encoder/resolve/main/encoder_simclr.pt"
    CLASSIFIER_URL = "https://huggingface.co/riad300/fish-simclr-encoder/resolve/main/fish_expert_weights.pt"
    
    # এনকোডার লোড
    encoder = get_encoder()
    e_state = torch.hub.load_state_dict_from_url(ENCODER_URL, map_location=device, check_hash=False)
    encoder.load_state_dict(e_state)
    
    # ক্লাসিফায়ার লোড (২১টি প্রজাতির জন্য)
    classifier = nn.Linear(2048, 21)
    c_state = torch.hub.load_state_dict_from_url(CLASSIFIER_URL, map_location=device, check_hash=False)
    
    # কী-ম্যাপিং ফিক্স: fc.weight কে weight এ রূপান্তর
    fixed_state = {k.replace('fc.', ''): v for k, v in c_state.items()}
    classifier.load_state_dict(fixed_state)
    
    encoder.eval()
    classifier.eval()
    return encoder, classifier

# --- ৩. ইউজার ইন্টারফেস সেটআপ ---
st.set_page_config(page_title="Expert BD Fish AI", layout="centered")
st.title("🐟 দেশি মাছ শনাক্তকারী (Expert Mode)")
st.write("আপনার ট্রেনিং করা মডেল দিয়ে নিখুঁতভাবে মাছ শনাক্ত করুন।")

# মডেল কল করা
try:
    encoder, classifier = load_expert_model()
    st.sidebar.success("মডেল এখন ১০০% রেডি!")
except Exception as e:
    st.error(f"মডেল লোডিং এরর: {e}")

# আপনার ফোল্ডার লিস্ট অনুযায়ী সঠিক নামের তালিকা
CLASSES = [
    "Baim (বাইন)", "Bata (বাটা)", "Batasio/Tengra (টেংরা)", "Chitul (চিতল)", 
    "Croaker/Poya (পোয়া)", "Hilsha (ইলিশ)", "Kajoli (কাজলী)", "Meni (মেনি)", 
    "Pabda (পাবদা)", "Poli (ফলি)", "Puti (পুঁটি)", "Rita (রিটা)", 
    "Rui (রুই)", "Rupchanda (রূপচাঁদা)", "Silver Carp (সিলভার কার্প)", 
    "Telapiya (তেলাপিয়া)", "Carp (কার্প)", "Koi (কৈ)", 
    "Kaikka (কাইক্কা)", "Koral (কোরাল)", "Shrimp (চিংড়ি)"
]

# --- ৪. ছবি আপলোড ও প্রেডিকশন ---
file = st.file_uploader("মাছের পরিষ্কার ছবি দিন", type=["jpg", "png", "jpeg"])

if file:
    img = Image.open(file).convert("RGB")
    st.image(img, use_container_width=True)
    
    # প্রসেসিং (আপনার ট্রেনিং এর স্ট্যান্ডার্ড অনুযায়ী)
    tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    with torch.no_grad():
        features = encoder(tf(img).unsqueeze(0))
        output = classifier(features)
        prob, idx = torch.max(torch.softmax(output, dim=1), 1)
    
    # ফলাফল দেখানো
    confidence = prob.item() * 100
    
    # যদি ১৬.৩৪% এর মতো কম কনফিডেন্স আসে তবে সতর্ক করবে
    if confidence < 30:
        st.warning(f"মডেল নিশ্চিত নয় (নিশ্চয়তা: {confidence:.2f}%)। দয়া করে পরিষ্কার ছবি দিন।")
    
    st.success(f"### শনাক্ত করা হয়েছে: **{CLASSES[idx.item()]}**")
    st.info(f"নিশ্চয়তা (Confidence): **{confidence:.2f}%**")

st.divider()
st.caption("টিপস: যদি রেজাল্ট ভুল আসে, তবে অ্যাপ মেনু থেকে 'Clear Cache' দিয়ে রিবুট করুন।")
