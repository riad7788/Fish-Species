import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import os
import requests

# ১. নোটবুক অনুযায়ী আর্কিটেকচার
class SimCLR_Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        base_model = models.resnet18(weights=None)
        self.features = nn.Sequential(*list(base_model.children())[:-1])

    def forward(self, x):
        x = self.features(x)
        return x.view(x.size(0), -1)

class Classifier(nn.Module):
    def __init__(self, in_dim, num_classes):
        super().__init__()
        self.fc = nn.Linear(in_dim, num_classes)

    def forward(self, x):
        return self.fc(x)

# ২. মডেল লোড করার প্রফেশনাল ফাংশন
@st.cache_resource
def load_full_system():
    device = torch.device("cpu")
    
    # এনকোডার লিঙ্ক (আপনার দেওয়া Hugging Face লিঙ্ক)
    ENCODER_URL = "https://huggingface.co/riad300/fish-simclr-encoder/resolve/main/encoder_simclr.pt"
    
    # এনকোডার ডাউনলোড ও লোড
    encoder = SimCLR_Encoder()
    encoder_state = torch.hub.load_state_dict_from_url(ENCODER_URL, map_location=device)
    encoder.load_state_dict(encoder_state)
    
    # ক্লাসিফায়ার লোড (এটি আপনার GitHub এর models/classifier.pt থেকে আসবে)
    base_dir = os.path.dirname(os.path.abspath(__file__))
    classifier_path = os.path.join(base_dir, "models", "classifier.pt")
    
    classifier = Classifier(512, 21) # আপনার নোটবুক অনুযায়ী ২১টি ক্লাস
    
    if os.path.exists(classifier_path):
        # weights_only=False দিয়ে লোড করা হচ্ছে কারণ আপনি পুরো অবজেক্ট সেভ করেছেন
        checkpoint = torch.load(classifier_path, map_location=device, weights_only=False)
        if isinstance(checkpoint, dict):
            classifier.load_state_dict(checkpoint)
        else:
            classifier = checkpoint
            
    encoder.eval()
    classifier.eval()
    return encoder, classifier

# ৩. ইউজার ইন্টারফেস
st.set_page_config(page_title="Fish AI Expert", layout="centered")
st.title("🐟 Fish Species Classification (SimCLR)")

with st.spinner('AI Models loading... Please wait.'):
    encoder, classifier = load_full_system()

uploaded_file = st.file_uploader("Upload fish image", type=["jpg", "png", "jpeg"])

if uploaded_file:
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, use_container_width=True)
    
    # নোটবুক অনুযায়ী ট্রান্সফর্ম
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    input_tensor = preprocess(img).unsqueeze(0)
    
    with torch.no_grad():
        feats = encoder(input_tensor)
        preds = classifier(feats)
        idx = torch.argmax(preds, 1).item()
    
    # আপনার নোটবুকের ২১টি মাছের নাম
    CLASSES = ["Biam", "Bata", "Batasio(tenra)", "Chitul", "Croaker(Poya)", "Hilsha",
               "Kajoli", "Meni", "Pabda", "Poli", "Puti", "Rita", "Rui", "Rupchanda",
               "Silver Carp", "Telapiya", "carp", "Koi", "kaikka", "koral", "shrimp"]
    
    st.success(f"### Predicted Species: {CLASSES[idx]}")
