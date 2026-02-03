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
        # ResNet50 বেস মডেল
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

# --- ২. লোডিং ফাংশন (যেকোনো ফরম্যাট হ্যান্ডেল করবে) ---
@st.cache_resource
def load_models():
    device = torch.device("cpu")
    
    # এনকোডার লোড (Hugging Face)
    encoder = SimCLR_Encoder()
    ENCODER_URL = "https://huggingface.co/riad300/fish-simclr-encoder/resolve/main/encoder_simclr.pt"
    
    try:
        e_state = torch.hub.load_state_dict_from_url(ENCODER_URL, map_location=device)
        encoder.load_state_dict(e_state)
    except Exception as e:
        st.error(f"Encoder Load Error: {e}")

    # ক্লাসিফায়ার লোড (GitHub)
    classifier = Classifier()
    base_dir = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(base_dir, "models", "classifier.pt")
    
    if os.path.exists(path):
        try:
            # weights_only=False দিয়ে লোড করা হচ্ছে যাতে 'MARK' এরর না আসে
            checkpoint = torch.load(path, map_location=device, weights_only=False)
            
            # যদি ফাইলটি শুধু state_dict হয়
            if isinstance(checkpoint, dict):
                if 'state_dict' in checkpoint:
                    classifier.load_state_dict(checkpoint['state_dict'])
                else:
                    classifier.load_state_dict(checkpoint)
            # যদি ফাইলটি পুরো মডেল অবজেক্ট হয়
            else:
                classifier = checkpoint
        except Exception as e:
            st.error(f"Classifier Load Error: {e}")
    
    encoder.eval()
    classifier.eval()
    return encoder, classifier

# --- ৩. ইউজার ইন্টারফেস ---
st.title("🐟 Fish Species AI Classifier")

encoder, classifier = load_models()

# আপনার নোটবুকের ২১টি মাছের সঠিক নাম
CLASSES = [
    "Biam", "Bata", "Batasio(tenra)", "Chitul", "Croaker(Poya)", "Hilsha",
    "Kajoli", "Meni", "Pabda", "Poli", "Puti", "Rita", "Rui", "Rupchanda",
    "Silver Carp", "Telapiya", "carp", "Koi", "kaikka", "koral", "shrimp"
]

file = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])

if file:
    img = Image.open(file).convert("RGB")
    st.image(img, width=300)
    
    # ট্রান্সফর্ম
    tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    with torch.no_grad():
        feats = encoder(tf(img).unsqueeze(0))
        out = classifier(feats)
        prob, idx = torch.max(torch.softmax(out, dim=1), 1)
    
    st.success(f"Result: {CLASSES[idx.item()]} (Confidence: {prob.item():.2%})")
