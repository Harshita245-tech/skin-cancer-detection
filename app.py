import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Load model
from tensorflow.keras.models import load_model

model = load_model("mobilenet_model.h5", compile=False)

class_names = ['akiec','bcc','bkl','df','mel','nv','vasc']

full_names = {
    'akiec': 'Actinic keratoses',
    'bcc': 'Basal cell carcinoma',
    'bkl': 'Benign keratosis',
    'df': 'Dermatofibroma',
    'mel': 'Melanoma',
    'nv': 'Melanocytic nevus',
    'vasc': 'Vascular lesion'
}

st.set_page_config(page_title="Skin Cancer Detection", layout="centered")

# 🎨 HEADER
st.markdown("""
<h1 style='text-align:center; color:#ff4b4b;'>🧬 Skin Cancer Detection</h1>
<p style='text-align:center;'>AI-powered detection system</p>
""", unsafe_allow_html=True)

st.markdown("---")

# 👤 PATIENT DETAILS
st.markdown("## 👤 Patient Details")

col1, col2 = st.columns(2)

with col1:
    name = st.text_input("Name")

with col2:
    age = st.number_input("Age", 1, 120)

gender = st.selectbox("Gender", ["Male", "Female", "Other"])

st.markdown("---")

# 📤 MULTIPLE IMAGE UPLOAD
st.markdown("## 📤 Upload Skin Images")

files = st.file_uploader(
    "Upload one or more images",
    type=["jpg","png","jpeg"],
    accept_multiple_files=True
)

def preprocess(img):
    img = img.resize((224,224))
    img = np.array(img)/255.0
    img = np.expand_dims(img, axis=0)
    return img

if files:
    if st.button("🔍 Predict All Images", use_container_width=True):

        for file in files:
            image = Image.open(file)

            st.markdown("---")
            st.image(image, caption=file.name, use_container_width=True)

            img = preprocess(image)
            pred = model.predict(img)[0]

            idx = np.argmax(pred)
            conf = pred[idx]

            disease = full_names[class_names[idx]]

            # 🎯 RESULT BOX
            st.markdown(f"""
            <div style="background-color:#1c1f26; padding:15px; border-radius:10px;">
                <h3 style="color:#00ff99;">🩺 {disease}</h3>
                <p>Confidence: {conf*100:.2f}%</p>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("---")

        # 👨‍⚕️ DOCTOR SUGGESTION
        st.markdown("## 👨‍⚕️ Doctor Suggestion")

        st.warning("⚠️ Consult a dermatologist.")
        st.info("ℹ️ Do not rely only on AI.")
        st.success("✅ Early detection is important.")

        # 📥 DOWNLOAD REPORT
        report = f"""
        SKIN CANCER REPORT

        Name: {name}
        Age: {age}
        Gender: {gender}

        Total Images: {len(files)}

        Note:
        Please consult doctor for confirmation.
        """

        st.download_button(
            "📥 Download Report",
            report,
            file_name="report.txt"
        )