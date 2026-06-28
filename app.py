
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# ── Konfigurasi halaman ──
st.set_page_config(
    page_title="Brain Tumor Classification",
    page_icon="🧠",
    layout="centered"
)

# ── Load Model ──
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model("best_model.keras")
    return model

model = load_model()

# ── Header ──
st.title("🧠 Brain Tumor Classification")
st.markdown("""
**Aplikasi klasifikasi tumor otak berbasis Deep Learning**
menggunakan model Hybrid CNN + MobileNetV2.

> Upload gambar MRI otak, model akan memprediksi apakah
> terdapat tumor atau tidak.
""")

st.divider()

# ── Upload Gambar ──
uploaded_file = st.file_uploader(
    "📤 Upload Gambar MRI Otak",
    type=["jpg", "jpeg", "png"],
    help="Format yang didukung: JPG, JPEG, PNG"
)

if uploaded_file is not None:
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Gambar Input")
        img = Image.open(uploaded_file).convert("RGB")
        st.image(img, use_column_width=True)

    # Preprocessing
    img_resized = img.resize((224, 224))
    img_array  = np.array(img_resized) / 255.0
    img_array  = np.expand_dims(img_array, axis=0)

    # Prediksi
    with st.spinner("Menganalisis gambar..."):
        pred_prob = model.predict(img_array)[0][0]

    label      = "TUMOR" if pred_prob > 0.5 else "NO TUMOR"
    confidence = pred_prob if pred_prob > 0.5 else 1 - pred_prob

    with col2:
        st.subheader("Hasil Prediksi")
        if label == "TUMOR":
            st.error(f"🔴 {label}")
        else:
            st.success(f"🟢 {label}")

        st.metric("Confidence", f"{confidence*100:.2f}%")
        st.progress(float(confidence))

        st.divider()
        st.markdown("**Detail Probabilitas:**")
        st.write(f"- Tumor    : {pred_prob*100:.2f}%")
        st.write(f"- No Tumor : {(1-pred_prob)*100:.2f}%")

st.divider()
st.caption("Dibuat untuk keperluan penelitian | CNN + MobileNetV2")
