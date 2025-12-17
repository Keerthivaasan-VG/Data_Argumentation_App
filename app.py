import streamlit as st
import zipfile
import numpy as np
from PIL import Image
from tensorflow.keras import layers
from io import BytesIO
import tensorflow as tf

# ================= PAGE CONFIG =================
st.set_page_config(
    page_title="Image Data Augmentation",
    page_icon="🧪",
    layout="centered"
)

st.markdown("<h1 style='text-align:center;'>🧪 Image Data Augmentation</h1>", unsafe_allow_html=True)
st.markdown("---")

# ================= INPUT =================
method = st.radio(
    "Choose image input method",
    ["Upload from File Manager", "Capture using Camera"]
)

uploaded_file = (
    st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])
    if method == "Upload from File Manager"
    else st.camera_input("Capture Image")
)

# ================= SETTINGS =================
num_images = st.slider("Number of augmented images", 1, 50, 50)

augmenter = tf.keras.Sequential([
    layers.RandomRotation(0.15),
    layers.RandomZoom(0.2),
    layers.RandomFlip("horizontal"),
    layers.RandomContrast(0.2)
])

# ================= PROCESS =================
if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Original Image", width="stretch")

    img = np.array(image)
    img = tf.expand_dims(img, axis=0)

    augmented_images = []

    for _ in range(num_images):
        aug = augmenter(img, training=True)
        aug_img = tf.cast(aug[0], tf.uint8).numpy()
        augmented_images.append(Image.fromarray(aug_img))

    # ================= PREVIEW =================
    st.markdown("### 🖼 Preview")
    cols = st.columns(4)
    for i, im in enumerate(augmented_images[:8]):
        cols[i % 4].image(im, width="stretch")

    # ================= ZIP DOWNLOAD =================
    zip_buffer = BytesIO()
    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as z:
        for i, im in enumerate(augmented_images):
            buf = BytesIO()
            im.save(buf, format="JPEG")
            z.writestr(f"augmented_{i+1}.jpg", buf.getvalue())

    st.download_button(
        "⬇ Download Augmented Images (ZIP)",
        data=zip_buffer.getvalue(),
        file_name="augmented_images.zip",
        mime="application/zip"
    )
