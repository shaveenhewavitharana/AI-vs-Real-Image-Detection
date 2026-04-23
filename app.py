import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
import cv2
import io

# =========================
# Load CSS
# =========================
def load_css():
    with open("style.css") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# =========================
# Config
# =========================
MODEL_PATH = "models/best_model.keras"
IMG_SIZE = (224, 224)
LAST_CONV_LAYER_NAME = "conv5_block3_out"

st.set_page_config(page_title="AI vs Real Detector", page_icon="🧠", layout="centered")

load_css()

# =========================
# Load Model
# =========================
@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)

model = load_model()

# =========================
# Grad-CAM
# =========================
def make_gradcam_heatmap(img_array, model, last_conv_layer_name):
    grad_model = tf.keras.models.Model(
        [model.inputs],
        [model.get_layer(last_conv_layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        loss = predictions[:, 0]

    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    heatmap = tf.maximum(heatmap, 0) / tf.reduce_max(heatmap)
    return heatmap.numpy()

def overlay_heatmap(heatmap, image):
    heatmap = cv2.resize(heatmap, (image.shape[1], image.shape[0]))
    heatmap = np.uint8(255 * heatmap)

    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    overlay = heatmap * 0.4 + image

    return np.uint8(overlay)

# =========================
# Sidebar
# =========================
st.sidebar.title("📌 About Project")
st.sidebar.markdown("""
### AI vs Real Detector

- 🧠 Deep Learning Model  
- 🔥 Grad-CAM Visualization  
- 📊 Confidence Scoring  

Built using:
- ResNet50
- TensorFlow
- Streamlit
""")

# =========================
# Title
# =========================
st.markdown("<h1>🧠 AI vs Real Image Detector</h1>", unsafe_allow_html=True)

st.markdown(
    "<p style='text-align:center;'>Upload an image to detect if it is AI-generated or real.</p>",
    unsafe_allow_html=True
)

st.markdown("---")

# =========================
# Upload
# =========================
uploaded_file = st.file_uploader("📤 Upload an Image", type=["jpg", "jpeg", "png", "webp"])

def preprocess(image):
    image = image.convert("RGB")
    image = image.resize(IMG_SIZE)
    img = np.array(image) / 255.0
    return np.expand_dims(img, axis=0)

# =========================
# Prediction
# =========================
if uploaded_file:
    image = Image.open(uploaded_file)

    st.image(image, caption="🖼 Uploaded Image", use_container_width=True)

    img_array = preprocess(image)
    prediction = model.predict(img_array)[0][0]

    if prediction > 0.5:
        label = "REAL"
        confidence = prediction
        color = "green"
    else:
        label = "AI GENERATED"
        confidence = 1 - prediction
        color = "red"

    st.markdown("---")
    st.subheader("🔍 Prediction Result")

    st.markdown(
        f"<h2 style='color:{color}; text-align:center;'>{label}</h2>",
        unsafe_allow_html=True
    )

    st.progress(float(confidence))
    st.info(f"Confidence: {confidence * 100:.2f}%")

    # =========================
    # Grad-CAM
    # =========================
    st.markdown("### 🔥 Grad-CAM Visualization")

    heatmap = make_gradcam_heatmap(img_array, model, LAST_CONV_LAYER_NAME)

    original = np.array(image.resize(IMG_SIZE))
    overlay = overlay_heatmap(heatmap, original)

    col1, col2 = st.columns(2)

    with col1:
        st.image(original, caption="Original", use_container_width=True)

    with col2:
        st.image(overlay, caption="Model Attention", use_container_width=True)

    # =========================
    # Explanation
    # =========================
    st.markdown("### 🧠 What this means")

    st.write("""
    - 🔴 Red → Important regions  
    - 🟡 Yellow → Medium importance  
    - 🔵 Blue → Less important  
    """)

    if label == "AI GENERATED":
        st.warning("The model detected patterns that may indicate AI-generated content.")
    else:
        st.success("The model detected natural patterns typical of real images.")

    # =========================
    # Download Button
    # =========================
    buf = io.BytesIO()
    Image.fromarray(overlay).save(buf, format="PNG")

    st.download_button(
        label="📥 Download Grad-CAM Result",
        data=buf.getvalue(),
        file_name="gradcam_result.png",
        mime="image/png"
    )

    st.markdown("---")

# =========================
# Footer
# =========================
st.markdown(
    "<p style='text-align:center; font-size:14px;'>Built with ❤️ using Deep Learning</p>",
    unsafe_allow_html=True
)