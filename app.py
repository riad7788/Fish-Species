import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

# --- ১. এনকোডার (হুবহু নোটবুক অনুযায়ী) ---
def get_encoder():
    encoder = models.resnet50(weights=None)
    encoder.fc = nn.Identity() 
    return encoder

# --- ২. ক্লাসিফায়ার লোড করার সবচাইতে নিরাপদ পদ্ধতি ---
@st.cache_resource
def load_full_model():
    device = torch.device("cpu")
    
    # Hugging Face URLs
    ENCODER_URL = "https://huggingface.co/riad300/fish-simclr-encoder/resolve/main/encoder_simclr.pt"
    CLASSIFIER_URL = "https://huggingface.co/riad300/fish-simclr-encoder/resolve/main/classifier.pt"
    
    # এনকোডার লোড
    encoder = get_encoder()
    e_state = torch.hub.load_state_dict_from_url(ENCODER_URL, map_location=device, check_hash=False)
    encoder.load_state_dict(e_state)
    
    # ক্লাসিফায়ার লোড - নামের ঝামেলা এড়াতে সরাসরি Linear লেয়ারের state_dict হ্যান্ডেল করা
    classifier = nn.Linear(2048, 21)
    c_state = torch.hub.load_state_dict_from_url(CLASSIFIER_URL, map_location=device, check_hash=False)
    
    # আপনার ফাইলে 'fc.weight' এর বদলে শুধু 'weight' থাকতে পারে, তাই এই চেকটি জরুরি
    new_state_dict = {}
    for key, value in c_state.items():
        new_key = key.replace('fc.', '') # 'fc.weight' কে 'weight' বানানো হচ্ছে
        new_state_dict[new_key] = value
        
    classifier.load_state_dict(new_state_dict)
    
    encoder.eval()
    classifier.eval()
    return encoder, classifier

# --- ৩. ইউজার ইন্টারফেস ---
st.set_page_config(page_title="Fish AI Expert")
st.title("🐟 Fish Species AI Classifier")

encoder, classifier = load_full_model()

# আপনার নোটবুকের ২১টি মাছের নামের লিস্ট
CLASSES = [
    "Biam", "Bata", "Batasio(tenra)", "Chitul", "Croaker(Poya)", "Hilsha",
    "Kajoli", "Meni", "Pabda", "Poli", "Puti", "Rita", "Rui", "Rupchanda",
    "Silver Carp", "Telapiya", "carp", "Koi", "kaikka", "koral", "shrimp"
]

uploaded_file = st.file_uploader("মাছের ছবি আপলোড করুন", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, width=400)
    
    # ইমেজ প্রি-প্রসেসিং (Normalization values are standard for ResNet)
    tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    with torch.no_grad():
        input_tensor = tf(image).unsqueeze(0)
        # ১. ফিচার বের করা
        features = encoder(input_tensor)
        # ২. সঠিক মাছ শনাক্ত করা
        outputs = classifier(features)
        probs = torch.softmax(outputs, dim=1)
        confidence, idx = torch.max(probs, 1)
    
    # রেজাল্ট ডিসপ্লে
    result_name = CLASSES[idx.item()]
    st.success(f"### রেজাল্ট: {result_name}")
    st.write(f"কনফিডেন্স: {confidence.item():.2%}")
