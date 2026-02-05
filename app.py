import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import requests
from io import BytesIO

# --- ১. এনকোডার ও মডেল লোডার (ফোর্স রিলোড সহ) ---
def get_encoder():
    encoder = models.resnet50(weights=None)
    encoder.fc = nn.Identity() 
    return encoder

@st.cache_resource(ttl=1)
def load_expert_model_v3():
    device = torch.device("cpu")
    ENCODER_URL = "https://huggingface.co/riad300/fish-simclr-encoder/resolve/main/encoder_simclr.pt"
    CLASSIFIER_URL = "https://huggingface.co/riad300/fish-simclr-encoder/resolve/main/fish_expert_weights.pt"
    
    encoder = get_encoder()
    e_state = torch.hub.load_state_dict_from_url(ENCODER_URL, map_location=device)
    encoder.load_state_dict(e_state)
    
    classifier = nn.Linear(2048, 21) # আপনার ২১টি মাছের জন্য
    response = requests.get(CLASSIFIER_URL)
    c_state = torch.load(BytesIO(response.content), map_location=device)
    
    # Key Mapping Fix
    fixed_state = {k.replace('fc.', ''): v for k, v in c_state.items()}
    classifier.load_state_dict(fixed_state)
    
    encoder.eval()
    classifier.eval()
    return encoder, classifier

# --- ২. ইউজার ইন্টারফেস ---
st.set_page_config(page_title="BD Fish Expert AI", layout="centered")
st.title("🐟 দেশি মাছ শনাক্তকারী (Pro Mode)")
st.info("টিপস: ২০০ ইপোকের মডেলটি যদি ভুল করে, তবে নিচের 'অন্যান্য সম্ভাবনা' দেখুন।")

# ২১টি মাছের সঠিক তালিকা
CLASSES = [
    "Baim (বাইন)", "Bata (বাটা)", "Batasio/Tengra (টেংরা)", "Chitul (চিতল)", 
    "Croaker/Poya (পোয়া)", "Hilsha (ইলিশ)", "Kajoli (কাজলী)", "Meni (মেনি)", 
    "Pabda (পাবদা)", "Poli (ফলি)", "Puti (পুঁটি)", "Rita (রিটা)", 
    "Rui (রুই)", "Rupchanda (রূপচাঁদা)", "Silver Carp (সিলভার কার্প)", 
    "Telapiya (তেলাপিয়া)", "Carp (কার্প)", "Koi (কৈ)", 
    "Kaikka (কাইক্কা)", "Koral (কোরাল)", "Shrimp (চিংড়ি)"
]

encoder, classifier = load_expert_model_v3()
file = st.file_uploader("মাছের ছবি আপলোড করুন", type=["jpg", "png", "jpeg"])

if file:
    img = Image.open(file).convert("RGB")
    st.image(img, use_container_width=True)
    
    tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    with torch.no_grad():
        features = encoder(tf(img).unsqueeze(0))
        output = classifier(features)
        # টপ ৩টি সম্ভাবনা বের করা
        probs, indices = torch.topk(torch.softmax(output, dim=1), 3)

    # রেজাল্ট ডিসপ্লে
    top_conf = probs[0][0].item() * 100
    top_label = CLASSES[indices[0][0].item()]

    if top_conf < 30: # ২০.৩৩% এর মতো কম রেজাল্টের জন্য সতর্কবার্তা
        st.warning(f"মডেল পুরোপুরি নিশ্চিত নয় (নিশ্চয়তা: {top_conf:.2f}%)")
    
    st.success(f"### প্রধান শনাক্তকরণ: **{top_label}**")
    st.progress(top_conf / 100)

    # অন্যান্য সম্ভাবনা (এটি আপনাকে সাহায্য করবে যদি প্রধান রেজাল্ট ভুল হয়)
    st.write("---")
    st.write("🔍 **অন্যান্য সম্ভাবনা:**")
    for i in range(1, 3):
        conf = probs[0][i].item() * 100
        label = CLASSES[indices[0][i].item()]
        st.write(f"{label}: {conf:.2f}%")
        st.progress(conf / 100)

if st.button('অ্যাপ রিফ্রেশ করুন (ক্যাশ ক্লিয়ার)'):
    st.cache_resource.clear()
    st.rerun()
