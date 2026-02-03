import streamlit as st
import os
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

# -----------------------
# ১. মডেল স্ট্রাকচার (আপনার নোটবুক অনুযায়ী)
# -----------------------
class SimCLR_Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        # নোটবুক অনুযায়ী ResNet18 এর লাস্ট লেয়ার বাদ দিয়ে ফিচার এক্সট্রাক্টর
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

# -----------------------
# ২. ক্লাস লিস্ট (২১টি মাছের নাম)
# -----------------------
CLASS_NAMES = [
    "Biam", "Bata", "Batasio(tenra)", "Chitul", "Croaker(Poya)", "Hilsha",
    "Kajoli", "Meni", "Pabda", "Poli", "Puti", "Rita", "Rui", "Rupchanda",
    "Silver Carp", "Telapiya", "carp", "Koi", "kaikka", "koral", "shrimp"
]

@st.cache_resource
def load_models():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    base_dir = os.path.dirname(os.path.abspath(__file__))
    
    # GitHub অনুযায়ী models ফোল্ডারের ভেতর classifier.pt
    classifier_path = os.path.join(base_dir, "models", "classifier.pt")

    if not os.path.exists(classifier_path):
        st.error(f"Model file not found at: {classifier_path}")
        st.stop()

    # আপনার নোটবুক অনুযায়ী ২১টি ক্লাসের জন্য মডেল সেটআপ
    encoder = SimCLR_Encoder()
    classifier = Classifier(512, len(CLASS_NAMES))

    try:
        # weights_only=False দিতে হবে কারণ আপনি পুরো অবজেক্ট সেভ করেছেন
        checkpoint = torch.load(classifier_path, map_location=device, weights_only=False)
        
        # আপনার সেভ করার ধরন অনুযায়ী লোড করা
        if isinstance(checkpoint, dict):
            classifier.load_state_dict(checkpoint)
        else:
            classifier = checkpoint
            
    except Exception as e:
        st.error(f"Error loading model: {e}")
        st.stop()

    encoder.to(device).eval()
    classifier.to(device).eval()
    return encoder, classifier, device

# -----------------------
# ৩. ইউজার ইন্টারফেস ও প্রেডিকশন
# -----------------------
st.set_page_config(page_title="Fish Classification", page_icon="🐟")
st.title("🐟 Fish Species Detection System (21 Species)")

encoder, classifier, device = load_models()

uploaded_file = st.file_uploader("Upload a fish image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)
    
    # নোটবুক অনুযায়ী ইমেজ ট্রান্সফর্ম
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    img_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        features = encoder(img_tensor)
        outputs = classifier(features)
        probs = torch.softmax(outputs, dim=1)
        pred_idx = torch.argmax(probs, dim=1).item()
        confidence = probs[0][pred_idx].item()

    st.success(f"### Predicted Species: {CLASS_NAMES[pred_idx]}")
    st.info(f"**Confidence Level:** {confidence:.2%}")
