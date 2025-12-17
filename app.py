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

# ================= SESSION STATE =================
if "augmented_images" not in st.session_state:
    st.session_state.augmented_images = None
if "zip_data" not in st.session_state:
    st.session_state.zip_data = None

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

num_images = st.slider("Number of augmented images", 1, 50, 50)

augmenter = tf.keras.Sequential([
    layers.RandomRotation(0.15),
    layers.RandomZoom(0.2),
    layers.RandomFlip("horizontal"),
    layers.RandomContrast(0.2)
])

# ================= BUTTON =================
if st.button("Generate Augmented Images") and uploaded_file:

    try:
        image = Image.open(uploaded_file).convert("RGB")
        image.thumbnail((512, 512))
        img = tf.expand_dims(np.array(image), axis=0)

        augmented_images = []

        for _ in range(num_images):
            aug = augmenter(img, training=True)
            aug_img = tf.cast(aug[0], tf.uint8).numpy()
            augmented_images.append(Image.fromarray(aug_img))

        st.session_state.augmented_images = augmented_images

        zip_buffer = BytesIO()
        with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as z:
            for i, im in enumerate(augmented_images):
                buf = BytesIO()
                im.save(buf, format="JPEG")
                z.writestr(f"augmented_{i+1}.jpg", buf.getvalue())

        zip_buffer.seek(0)
        st.session_state.zip_data = zip_buffer.getvalue()

        st.success("Images generated successfully!")

    except Exception as e:
        st.error("Something went wrong. Please upload a valid image.")

# ================= DISPLAY =================
if st.session_state.augmented_images:
    st.markdown("### 🖼 Preview")
    cols = st.columns(4)
    for i, im in enumerate(st.session_state.augmented_images[:8]):
        cols[i % 4].image(im, width="stretch")

    st.download_button(
        "⬇ Download Augmented Images (ZIP)",
        data=st.session_state.zip_data,
        file_name="augmented_images.zip",
        mime="application/zip"
    )
