import streamlit as st
from transformers import pipeline
from PIL import Image

# ১. প্রফেশনাল মডেল লোড করা (এটি Google-এর ViT মডেল)
@st.cache_resource
def load_professional_model():
    # এই মডেলটি কয়েক হাজার ক্যাটাগরি শনাক্ত করতে পারে
    return pipeline("image-classification", model="google/vit-base-patch16-224")

st.set_page_config(page_title="Fish AI Expert", page_icon="🐟")
st.title("🐟 Professional Fish Species Classifier")
st.write("বিশ্বমানের AI ব্যবহার করে যেকোনো মাছ শনাক্ত করুন।")

# মডেল কল করা
with st.spinner('AI মডেল তৈরি হচ্ছে... প্রথমবার ১-২ মিনিট সময় লাগতে পারে।'):
    classifier = load_professional_model()

# ২. ছবি আপলোড ইন্টারফেস
uploaded_file = st.file_uploader("একটি মাছের ছবি আপলোড করুন", type=["jpg", "png", "jpeg"])

if uploaded_file:
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Uploaded Image", use_container_width=True)
    
    # প্রেডিকশন বা মাছ শনাক্তকরণ
    with st.spinner('AI মাছটি বিশ্লেষণ করছে...'):
        results = classifier(img)
    
    st.success("### শনাক্ত করা ফলাফল:")
    
    # রেজাল্ট ডিসপ্লে
    for res in results:
        label = res['label']
        score = res['score']
        
        # প্রফেশনাল বার দিয়ে রেজাল্ট দেখানো
        st.write(f"**{label}** ({score*100:.2f}%)")
        st.progress(score)
