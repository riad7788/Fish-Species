import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

# --- ১. মডেল আর্কিটেকচার (আপনার নোটবুক অনুযায়ী হুবহু) ---
def get_encoder():
    # সরাসরি ResNet50 বেস মডেল
    encoder = models.resnet50(weights=None)
    # আপনার নোটবুক অনুযায়ী fc লেয়ারকে Identity করা হয়েছে
    encoder.fc = nn.Identity() 
    return encoder

class Classifier(nn.Module):
    def __init__(self, in_dim=2048, num_classes=21):
        super().__init__()
        self.fc = nn.Linear(in_dim, num_classes)

    def forward(self, x):
        return self.fc(x)

# --- ২. মডেল লোডিং (সরাসরি লিঙ্ক থেকে) ---
@st.cache_resource
def load_full_model():
    device = torch.device("cpu")
    
    # আপনার Hugging Face লিঙ্কসমূহ
    ENCODER_URL = "https://huggingface.co/riad300/fish-simclr-encoder/resolve/main/encoder_simclr.pt"
    CLASSIFIER_URL = "https://huggingface.co/riad300/fish-simclr-encoder/resolve/main/classifier.pt"
    
    # এনকোডার লোড
    encoder = get_encoder()
    e_state = torch.hub.load_state_dict_from_url(ENCODER_URL, map_location=device, check_hash=False)
    
    # "Missing key" এরর এড়াতে সরাসরি state_dict লোড
    encoder.load_state_dict(e_state)
    
    # ক্লাসিফায়ার লোড
    classifier = Classifier()
    c_state = torch.hub.load_state_dict_from_url(CLASSIFIER_URL, map_location=device, check_hash=False)
    
    if isinstance(c_state, dict):
        classifier.load_state_dict(c_state)
    else:
        classifier = c_state
        
    encoder.eval()
    classifier.eval()
    return encoder, classifier

# --- ৩. ইউজার ইন্টারফেস ---
st.set_page_config(page_title="Fish Species AI", page_icon="🐟")
st.title("🐟 Fish Species AI Classifier")

# মডেল কল করা
try:
    encoder, classifier = load_full_model()
    st.sidebar.success("মডেল লোড হয়েছে!")
except Exception as e:
    st.error(f"লোডিং এরর: {e}")

# মাছের নামসমূহ
CLASSES = [
    "Biam", "Bata", "Batasio(tenra)", "Chitul", "Croaker(Poya)", "Hilsha",
    "Kajoli", "Meni", "Pabda", "Poli", "Puti", "Rita", "Rui", "Rupchanda",
    "Silver Carp", "Telapiya", "carp", "Koi", "kaikka", "koral", "shrimp"
]

uploaded_file = st.file_uploader("একটি মাছের ছবি আপলোড করুন", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)
    
    # ইমেজ প্রি-প্রসেসিং
    tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    input_tensor = tf(image).unsqueeze(0)
    
    with torch.no_grad():
        features = encoder(input_tensor)
        outputs = classifier(features)
        probs = torch.softmax(outputs, dim=1)
        confidence, idx = torch.max(probs, 1)
    
    st.success(f"### শনাক্ত করা মাছ: {CLASSES[idx.item()]}")
    st.info(f"কনফিডেন্স: {confidence.item():.2%}")
