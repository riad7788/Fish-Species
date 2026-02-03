import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

# --- ১. সঠিক আর্কিটেকচার ---
def get_encoder():
    encoder = models.resnet50(weights=None)
    encoder.fc = nn.Identity() 
    return encoder

# --- ২. ক্যাশ ক্লিয়ারিং সাপোর্টেড লোডার ---
@st.cache_resource(show_spinner=True)
def load_full_model():
    device = torch.device("cpu")
    
    # আপনার নতুন লিঙ্কসমূহ
    ENCODER_URL = "https://huggingface.co/riad300/fish-simclr-encoder/resolve/main/encoder_simclr.pt"
    CLASSIFIER_URL = "https://huggingface.co/riad300/fish-simclr-encoder/resolve/main/classifier_final.pt"
    
    # এনকোডার
    encoder = get_encoder()
    e_state = torch.hub.load_state_dict_from_url(ENCODER_URL, map_location=device, check_hash=False)
    encoder.load_state_dict(e_state)
    
    # ক্লাসিফায়ার - নোটবুকের ট্রেনিং অনুযায়ী
    classifier = nn.Linear(2048, 21)
    c_state = torch.hub.load_state_dict_from_url(CLASSIFIER_URL, map_location=device, check_hash=False)
    
    # কী-ম্যাপিং (fc.weight -> weight)
    fixed_state = {k.replace('fc.', ''): v for k, v in c_state.items()}
    classifier.load_state_dict(fixed_state)
    
    encoder.eval()
    classifier.eval()
    return encoder, classifier

# --- ৩. ইউজার ইন্টারফেস ---
st.set_page_config(page_title="Fish AI Expert", layout="centered")
st.title("🐟 Fish Species AI Classifier")

# মডেল কল করা
encoder, classifier = load_full_model()

# সঠিক ক্লাসের লিস্ট
CLASSES = [
    "Biam", "Bata", "Batasio(tenra)", "Chitul", "Croaker(Poya)", "Hilsha",
    "Kajoli", "Meni", "Pabda", "Poli", "Puti", "Rita", "Rui", "Rupchanda",
    "Silver Carp", "Telapiya", "carp", "Koi", "kaikka", "koral", "shrimp"
]

file = st.file_uploader("মাছের ছবি দিন", type=["jpg", "png", "jpeg"])

if file:
    img = Image.open(file).convert("RGB")
    st.image(img, use_container_width=True)
    
    # প্রসেসিং
    tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    with torch.no_grad():
        feats = encoder(tf(img).unsqueeze(0))
        out = classifier(feats)
        prob, idx = torch.max(torch.softmax(out, dim=1), 1)
    
    # রেজাল্ট চেক (যদি কনফিডেন্স কম হয় তবে ওয়ার্নিং দেবে)
    st.success(f"### শনাক্ত করা হয়েছে: **{CLASSES[idx.item()]}**")
    st.info(f"কনফিডেন্স লেভেল: **{prob.item()*100:.2f}%**")
    
    if prob.item() < 0.30:
        st.warning("সতর্কতা: কনফিডেন্স খুব কম! সম্ভবত মডেলটি ভুল করছে। ক্যাশ ক্লিয়ার করে আবার চেষ্টা করুন।")
