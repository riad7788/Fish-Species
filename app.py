import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import requests
from io import BytesIO

# --- ১. ইউনিক মডেল লোডার (পুরনো ক্যাশ ডিলিট করার জন্য) ---
@st.cache_resource(ttl=1) # প্রতি ১ সেকেন্ড পর পর ক্যাশ চেক করবে
def load_expert_model_v2():
    device = torch.device("cpu")
    
    # নতুন এবং সঠিক লিঙ্ক
    CLASSIFIER_URL = "https://huggingface.co/riad300/fish-simclr-encoder/resolve/main/fish_expert_weights.pt"
    ENCODER_URL = "https://huggingface.co/riad300/fish-simclr-encoder/resolve/main/encoder_simclr.pt"
    
    # এনকোডার
    encoder = models.resnet50(weights=None)
    encoder.fc = nn.Identity()
    e_state = torch.hub.load_state_dict_from_url(ENCODER_URL, map_location=device)
    encoder.load_state_dict(e_state)
    
    # ক্লাসিফায়ার (আপনার ২১টি দেশি মাছের জন্য)
    classifier = nn.Linear(2048, 21)
    
    # সরাসরি বাইনারি ডাউনলোড করে লোড করা (যাতে কোনোভাবেই পুরনো ক্যাশ না থাকে)
    response = requests.get(CLASSIFIER_URL)
    c_state = torch.load(BytesIO(response.content), map_location=device)
    
    # কী-ম্যাপিং ফিক্স
    fixed_state = {k.replace('fc.', ''): v for k, v in c_state.items()}
    classifier.load_state_dict(fixed_state)
    
    encoder.eval()
    classifier.eval()
    return encoder, classifier

# --- ২. অ্যাপ ইন্টারফেস ---
st.set_page_config(page_title="Fish Expert Pro", layout="centered")
st.title("🐟 দেশি মাছ শনাক্তকারী (Final Fix)")

# ক্যাশ ক্লিয়ার বাটন (সরাসরি ইন্টারফেসে)
if st.button('অ্যাপ যদি ভুল রেজাল্ট দেয় তবে এখানে ক্লিক করুন (Force Refresh)'):
    st.cache_resource.clear()
    st.rerun()

encoder, classifier = load_expert_model_v2()

# ২১টি মাছের সঠিক নামের তালিকা
CLASSES = [
    "Baim (বাইন)", "Bata (বাটা)", "Batasio/Tengra (টেংরা)", "Chitul (চিতল)", 
    "Croaker/Poya (পোয়া)", "Hilsha (ইলিশ)", "Kajoli (কাজলী)", "Meni (মেনি)", 
    "Pabda (পাবদা)", "Poli (ফলি)", "Puti (পুঁটি)", "Rita (রিটা)", 
    "Rui (রুই)", "Rupchanda (রূপচাঁদা)", "Silver Carp (সিলভার কার্প)", 
    "Telapiya (তেলাপিয়া)", "Carp (কার্প)", "Koi (কৈ)", 
    "Kaikka (কাইkka)", "Koral (কোরাল)", "Shrimp (চিংড়ি)"
]

file = st.file_uploader("মাছের ছবি দিন", type=["jpg", "png", "jpeg"])

if file:
    img = Image.open(file).convert("RGB")
    st.image(img, use_container_width=True)
    
    tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    with torch.no_grad():
        feats = encoder(tf(img).unsqueeze(0))
        out = classifier(feats)
        prob, idx = torch.max(torch.softmax(out, dim=1), 1)
    
    # রেজাল্ট
    confidence = prob.item() * 100
    st.success(f"### শনাক্ত করা হয়েছে: **{CLASSES[idx.item()]}**")
    st.info(f"নিশ্চয়তা (Confidence): **{confidence:.2f}%**")
