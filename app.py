import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

# --- ১. মডেল স্ট্রাকচার (আপনার নোটবুক অনুযায়ী ResNet50) ---
class SimCLR_Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        # ResNet50 বেস মডেল
        base_model = models.resnet50(weights=None)
        self.encoder = nn.Sequential(*list(base_model.children())[:-1])

    def forward(self, x):
        h = self.encoder(x)
        return h.view(h.size(0), -1) # আউটপুট ডাইমেনশন: 2048

class Classifier(nn.Module):
    def __init__(self, in_dim=2048, num_classes=21):
        super().__init__()
        self.fc = nn.Linear(in_dim, num_classes)

    def forward(self, x):
        return self.fc(x)

# --- ২. সরাসরি লিঙ্ক থেকে মডেল লোডিং (Hugging Face) ---
@st.cache_resource
def load_full_model():
    device = torch.device("cpu")
    
    # এনকোডার এবং ক্লাসিফায়ার ইউআরএল
    ENCODER_URL = "https://huggingface.co/riad300/fish-simclr-encoder/resolve/main/encoder_simclr.pt"
    CLASSIFIER_URL = "https://huggingface.co/riad300/fish-simclr-encoder/resolve/main/classifier.pt"
    
    # এনকোডার লোড
    encoder = SimCLR_Encoder()
    e_state = torch.hub.load_state_dict_from_url(ENCODER_URL, map_location=device, check_hash=False)
    encoder.load_state_dict(e_state)
    
    # ক্লাসিফায়ার লোড
    classifier = Classifier()
    c_state = torch.hub.load_state_dict_from_url(CLASSIFIER_URL, map_location=device, check_hash=False)
    
    # যেহেতু আপনি নোটবুকে পুরো মডেল বা state_dict যেকোনোভাবে সেভ করতে পারেন, তাই এই চেকটি রাখা হয়েছে
    if isinstance(c_state, dict):
        classifier.load_state_dict(c_state)
    else:
        classifier = c_state
        
    encoder.eval()
    classifier.eval()
    return encoder, classifier

# --- ৩. ইউজার ইন্টারফেস (UI) ---
st.set_page_config(page_title="Fish Species AI", page_icon="🐟")
st.title("🐟 Fish Species AI Classifier")
st.markdown("২১টি প্রজাতির মাছ শনাক্ত করতে ছবি আপলোড করুন।")

# মডেল কল করা
with st.spinner('মডেল ডাউনলোড হচ্ছে... প্রথমবার কিছুক্ষণ সময় নিতে পারে।'):
    encoder, classifier = load_full_model()

# আপনার নোটবুকের ২১টি মাছের নাম
CLASSES = [
    "Biam", "Bata", "Batasio(tenra)", "Chitul", "Croaker(Poya)", "Hilsha",
    "Kajoli", "Meni", "Pabda", "Poli", "Puti", "Rita", "Rui", "Rupchanda",
    "Silver Carp", "Telapiya", "carp", "Koi", "kaikka", "koral", "shrimp"
]

uploaded_file = st.file_uploader("একটি ছবি সিলেক্ট করুন", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)
    
    # নোটবুক অনুযায়ী ইমেজ প্রসেসিং
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
    
    st.success(f"### রেজাল্ট: {CLASSES[idx.item()]}")
    st.info(f"কনফিডেন্স: {confidence.item():.2%}")
