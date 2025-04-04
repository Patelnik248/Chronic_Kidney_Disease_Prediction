import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import time

# Load models
model1 = tf.keras.models.load_model("./Models/CKD_Pred_Model_V_1.keras")
model2 = tf.keras.models.load_model("./Models/CKD_Pred_Model_V_2.keras")

def preprocess_image(image):
    image = image.convert('L')
    image = image.resize((200, 200))  # Resize to model input size
    image = np.array(image) / 255.0  # Normalize
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    return image

# Streamlit page config
st.set_page_config(page_title="Kidney Disease Prediction", layout="wide")

# Custom styling
st.markdown(
    """
    <style>
    body {
        background-color: #0d1117;
        color: white;
        font-family: 'Arial', sans-serif;
    }
    .stApp {
        text-align: center;
    }
    .stButton>button {
        background: linear-gradient(135deg, #ff416c, #ff4b2b);
        color: white;
        font-size: 18px;
        border-radius: 10px;
        padding: 12px 24px;
        font-weight: bold;
        transition: 0.3s;
        border: none;
    }
    .stButton>button:hover {
        background: linear-gradient(135deg, #ff4b2b, #ff416c);
        color: white !important;
    }
    .stButton>button:active {
        background: linear-gradient(135deg, #ff4b2b, #ff416c) !important;
        color: white !important;
    }
    .stImage img {
        border-radius: 15px;
        box-shadow: 0px 4px 15px rgba(255, 64, 64, 0.3);
    }
    .result-box {
        background: rgba(255, 255, 255, 0.1);
        padding: 20px;
        border-radius: 12px;
        text-align: center;
        box-shadow: 0px 4px 10px rgba(255, 255, 255, 0.2);
    }
    .prediction-header {
        font-size: 20px;
        font-weight: bold;
        color: #ff4b2b;
        margin-bottom: 10px;
    }
    .stButton>button:focus {
        color: white !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# App title and description
st.title("🔬 Kidney Disease Prediction")
st.markdown("### Upload a CT Scan Image for Analysis")

# File uploader
uploaded_file = st.file_uploader("Upload a CT Scan Image", type=["png", "jpg", "jpeg"], help="Supported formats: PNG, JPG, JPEG")

class_labels = ['Cyst', 'Normal', 'Stone', 'Tumor']

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", width=350)
    
    if st.button("🧪 Predict Now"):
        with st.spinner("Processing Image... Please wait."):
            time.sleep(1.5)  # Simulate loading time
            processed_image = preprocess_image(image)
            pred1 = model1.predict(processed_image)[0]
            pred2 = model2.predict(processed_image)[0]
        
        st.markdown("---")
        
        st.markdown("### 🧠 Prediction Results")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("<div class='prediction-header'>🔍 Model 1 Prediction</div>", unsafe_allow_html=True)
            st.markdown(f"<div class='result-box'><h3>{class_labels[np.argmax(pred1)]}</h3><p>Confidence: {max(pred1) * 100:.2f}%</p></div>", unsafe_allow_html=True)
        
        with col2:
            st.markdown("<div class='prediction-header'>🔍 Model 2 Prediction</div>", unsafe_allow_html=True)
            st.markdown(f"<div class='result-box'><h3>{class_labels[np.argmax(pred2)]}</h3><p>Confidence: {max(pred2) * 100:.2f}%</p></div>", unsafe_allow_html=True)
        
        st.markdown("---")
