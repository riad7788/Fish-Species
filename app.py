import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import os

# --- ১. সঠিক মডেল আর্কিটেকচার (ResNet50) ---
class SimCLR_Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        # নোটবুক অনুযায়ী ResNet50 বেস মডেল
        base_model = models.resnet50(weights=None)
        self.encoder = nn.Sequential(*list(base_model.children())[:-1])

    def forward(self, x):
        h = self.encoder(x)
        return h.view(h.size(0), -1)

class Classifier(nn.Module):
    def __init__(self, in_dim=2048, num_classes=21):
        super().__init__()
        self.fc = nn.Linear(in_dim, num_classes)

    def forward(self, x):
        return self.fc(x)

# --- ২. ১০০% এরর-ফ্রি লোডিং ফাংশন ---
@st.cache_resource
def load_models():
    device = torch.device("cpu")
    
    # এনকোডার লোড (Hugging Face থেকে সরাসরি)
    encoder = SimCLR_Encoder()
    ENCODER_URL = "https://huggingface.co/riad300/fish-simclr-encoder/resolve/main/encoder_simclr.pt"
    
    try:
        e_state = torch.hub.load_state_dict_from_url(ENCODER_URL, map_location=device, check_hash=False)
        encoder.load_state_dict(e_state)
    except Exception as e:
        st.error(f"Encoder Error: {e}")

    # ক্লাসিফায়ার লোড
    classifier = Classifier()
    # আপনার GitHub-এর পাথ
    path = os.path.join(os.getcwd(), "models", "classifier.pt")
    
    if os.path.exists(path):
        try:
            # weights_only=False এবং ম্যাপ লোকেশন ফিক্স করা হয়েছে
            checkpoint = torch.load(path, map_location=device, weights_only=False)
            if isinstance(checkpoint, dict):
                classifier.load_state_dict(checkpoint)
            else:
                classifier = checkpoint
        except Exception as e:
            st.warning("ফাইলটি করাপ্ট হয়েছে। আপনি কি Git LFS ব্যবহার করেছেন?")
            st.info("বিকল্প সমাধান: ফাইলটি আবার আপলোড করুন।")
    
    encoder.eval()
    classifier.eval()
    return encoder, classifier

# --- ৩. মেইন অ্যাপ UI ---
st.set_page_config(page_title="Fish AI Expert", layout="centered")
st.title("🐟 Fish Species Detection System")

encoder, classifier = load_models()

CLASSES = [
    "Biam", "Bata", "Batasio(tenra)", "Chitul", "Croaker(Poya)", "Hilsha",
    "Kajoli", "Meni", "Pabda", "Poli", "Puti", "Rita", "Rui", "Rupchanda",
    "Silver Carp", "Telapiya", "carp", "Koi", "kaikka", "koral", "shrimp"
]

uploaded_file = st.file_uploader("একটি ছবি আপলোড করুন", type=["jpg", "png", "jpeg"])

if uploaded_file:
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Uploaded Image", use_container_width=True)
    
    # নোটবুক অনুযায়ী ইমেজ প্রসেসিং
    tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    input_data = tf(img).unsqueeze(0)
    
    with torch.no_grad():
        features = encoder(input_data)
        output = classifier(features)
        prob, idx = torch.max(torch.softmax(output, dim=1), 1)
    
    st.success(f"### রেজাল্ট: {CLASSES[idx.item()]}")
    st.write(f"কনফিডেন্স: {prob.item():.2%}")
