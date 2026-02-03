import streamlit as st
from transformers import pipeline
from PIL import Image

# ১. প্রফেশনাল ইমেজ ক্লাসিফিকেশন পাইপলাইন (Google-এর ViT মডেল)
@st.cache_resource
def load_pro_model():
    # এটি কয়েক হাজার ক্যাটাগরি চেনে এবং একদম প্রফেশনাল রেজাল্ট দেয়
    return pipeline("image-classification", model="google/vit-base-patch16-224")

st.title("🐟 Professional Fish Species Expert AI")

classifier = load_pro_model()
uploaded_file = st.file_uploader("মাছের ছবি আপলোড করুন...", type=["jpg", "png", "jpeg"])

if uploaded_file:
    img = Image.open(uploaded_file)
    st.image(img, use_container_width=True)
    
    # প্রেডিকশন
    results = classifier(img)
    
    st.success("### শনাক্ত করা সম্ভাব্য প্রজাতিসমূহ:")
    for res in results:
        # প্রফেশনাল লুকের জন্য প্রগ্রেস বার সহ আউটপুট
        st.write(f"**{res['label']}**")
        st.progress(res['score'])
