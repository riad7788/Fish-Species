import streamlit as st
import os
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

# -----------------------
# ১. নোটবুক অনুযায়ী মডেল স্ট্রাকচার
# -----------------------
class SimCLR_Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        # নোটবুক অনুযায়ী ResNet18 ব্যবহার করা হয়েছে
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
# ২. ক্লাস লিস্ট (আপনার নোটবুক অনুযায়ী ২১টি)
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
    model_path = os.path.join(base_dir, "models", "classifier.pt")

    if not os.path.exists(model_path):
        st.error(f"Model file not found at: {model_path}")
        st.stop()

    # আপনার নোটবুক অনুযায়ী ইনপুট ডাইমেনশন ৫১২ এবং ক্লাস সংখ্যা ২১
    encoder = SimCLR_Encoder()
    classifier = Classifier(512, len(CLASS_NAMES))

    try:
        # weights_only=False দিয়ে লোড করা হচ্ছে কারণ আপনি পুরো অবজেক্ট সেভ করেছেন
        loaded_model = torch.load(model_path, map_location=device, weights_only=False)
        
        # চেক করা হচ্ছে এটি কি state_dict নাকি পুরো মডেল অবজেক্ট
        if isinstance(loaded_model, dict):
            classifier.load_state_dict(loaded_model)
        else:
            classifier = loaded_model
            
    except Exception as e:
        st.error(f"মডেল লোড করতে এরর: {e}")
        st.stop()

    encoder.to(device).eval()
    classifier.to(device).eval()
    return encoder, classifier, device

# -----------------------
# ৩. ইউজার ইন্টারফেস (UI)
# -----------------------
st.set_page_config(page_title="Fish Detection", page_icon="🐟")
st.title("🐟 Fish Species Detection System")
st.write("Upload a fish image to classify its species.")

encoder, classifier, device = load_models()

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption='Uploaded Image', use_container_width=True)
    
    # নোটবুকের ট্রান্সফর্ম লজিক অনুযায়ী
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    img_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        # নোটবুক অনুযায়ী ফিচার বের করে ক্লাসিফাই করা
        features = encoder(img_tensor)
        outputs = classifier(features)
        probs = torch.softmax(outputs, dim=1)
        pred_idx = torch.argmax(probs, dim=1).item()
        confidence = probs[0][pred_idx].item()

    st.success(f"### Prediction: {CLASS_NAMES[pred_idx]}")
    st.info(f"**Confidence:** {confidence:.2%}")
