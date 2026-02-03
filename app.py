import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import os

# --- ১. মডেল স্ট্রাকচার (আপনার নোটবুক অনুযায়ী ResNet50) ---
class SimCLR_Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        # ResNet50 বেস মডেল
        base_model = models.resnet50(weights=None)
        # লাস্ট লেয়ার বাদ দিয়ে ফিচার এক্সট্রাক্টর
        self.encoder = nn.Sequential(*list(base_model.children())[:-1])

    def forward(self, x):
        h = self.encoder(x)
        return h.view(h.size(0), -1) # আউটপুট ডাইমেনশন: 2048

class Classifier(nn.Module):
    def __init__(self, in_dim, num_classes):
        super().__init__()
        self.fc = nn.Linear(in_dim, num_classes)

    def forward(self, x):
        return self.fc(x)

# --- ২. মডেল লোডিং (অটোমেটিক এরর হ্যান্ডলিং সহ) ---
@st.cache_resource
def load_full_system():
    device = torch.device("cpu")
    
    # এনকোডার লোড (Hugging Face থেকে)
    ENCODER_URL = "https://huggingface.co/riad300/fish-simclr-encoder/resolve/main/encoder_simclr.pt"
    encoder = SimCLR_Encoder()
    
    try:
        # সরাসরি URL থেকে ওয়েটস লোড
        state_dict = torch.hub.load_state_dict_from_url(ENCODER_URL, map_location=device, check_hash=False)
        encoder.load_state_dict(state_dict)
    except Exception as e:
        st.error(f"এনকোডার লোড করতে সমস্যা: {e}")

    # ক্লাসিফায়ার লোড (GitHub এর models/classifier.pt থেকে)
    base_dir = os.path.dirname(os.path.abspath(__file__))
    classifier_path = os.path.join(base_dir, "models", "classifier.pt")
    
    # ResNet50 এর জন্য ইনপুট ২০৪৮ এবং মাছ ২১টি
    classifier = Classifier(2048, 21) 
    
    if os.path.exists(classifier_path):
        try:
            # weights_only=False দেওয়া হয়েছে যাতে কোনো Pickling এরর না আসে
            c_state = torch.load(classifier_path, map_location=device, weights_only=False)
            if isinstance(c_state, dict):
                classifier.load_state_dict(c_state)
            else:
                classifier = c_state
        except Exception as e:
            st.error(f"ক্লাসিফায়ার লোড করতে সমস্যা: {e}")
    else:
        st.error("Error: 'models/classifier.pt' ফাইলটি খুঁজে পাওয়া যায়নি। দয়া করে ফাইলটি আপলোড করুন।")
        
    encoder.eval()
    classifier.eval()
    return encoder, classifier

# --- ৩. মেইন ইউজার ইন্টারফেস ---
st.set_page_config(page_title="Fish AI Expert", page_icon="🐟")

st.title("🐟 প্রফেশনাল ফিশ ক্লাসিফিকেশন AI")
st.markdown("২১টি প্রজাতির মাছ নির্ভুলভাবে শনাক্ত করতে ছবি আপলোড করুন।")

# মডেল কল করা
encoder, classifier = load_full_system()

# ২১টি মাছের নামের লিস্ট
CLASSES = [
    "Biam", "Bata", "Batasio(tenra)", "Chitul", "Croaker(Poya)", "Hilsha",
    "Kajoli", "Meni", "Pabda", "Poli", "Puti", "Rita", "Rui", "Rupchanda",
    "Silver Carp", "Telapiya", "carp", "Koi", "kaikka", "koral", "shrimp"
]

uploaded_file = st.file_uploader("ছবি সিলেক্ট করুন...", type=["jpg", "png", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)
    
    # ইমেজ প্রসেসিং (Image Preprocessing)
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    input_tensor = preprocess(image).unsqueeze(0)
    
    with torch.no_grad():
        # ১. এনকোডার দিয়ে ফিচার বের করা
        features = encoder(input_tensor)
        # ২. ক্লাসিফায়ার দিয়ে রেজাল্ট বের করা
        outputs = classifier(features)
        probs = torch.softmax(outputs, dim=1)
        confidence, idx = torch.max(probs, 1)
    
    # রেজাল্ট ডিসপ্লে
    st.success(f"### শনাক্ত করা হয়েছে: **{CLASSES[idx.item()]}**")
    st.info(f"কনফিডেন্স লেভেল: **{confidence.item():.2% Prime}**")
