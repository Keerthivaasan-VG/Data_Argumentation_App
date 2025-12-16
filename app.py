import streamlit as st
import zipfile
import numpy as np
from PIL import Image
from tensorflow.keras import layers
from io import BytesIO

# ================= PAGE CONFIG =================
st.set_page_config(
    page_title="Image Data Augmentation",
    page_icon="🧪",
    layout="centered"
)

# ================= HEADER =================
st.markdown("<h1 style='text-align:center;'>🧪 Image Data Augmentation</h1>", unsafe_allow_html=True)
st.markdown("---")

# ================= INPUT =================
method = st.radio("Choose image input method",
                  ["Upload from File Manager", "Capture using Camera"])

uploaded_file = (
    st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])
    if method == "Upload from File Manager"
    else st.camera_input("Capture Image")
)

# ================= SETTINGS =================
num_images = st.slider("Number of augmented images", 1, 50, 20)

augmenter = layers.RandomRotation(0.15)
augmenter2 = layers.RandomZoom(0.2)
augmenter3 = layers.RandomFlip("horizontal")

# ================= PROCESS =================
if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Original Image", width="stretch")

    img = np.array(image)
    img = np.expand_dims(img, axis=0)

    augmented_images = []

    for _ in range(num_images):
        aug = augmenter(img)
        aug = augmenter2(aug)
        aug = augmenter3(aug)
        aug_img = Image.fromarray(aug[0].numpy().astype("uint8"))
        augmented_images.append(aug_img)

    # ================= PREVIEW =================
    st.markdown("### 🖼 Preview")
    cols = st.columns(4)
    for i, im in enumerate(augmented_images[:8]):
        cols[i % 4].image(im, width="stretch")

    # ================= ZIP =================
    zip_buffer = BytesIO()
    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as z:
        for i, im in enumerate(augmented_images):
            buf = BytesIO()
            im.save(buf, format="JPEG")
            z.writestr(f"augmented_{i+1}.jpg", buf.getvalue())

    st.download_button(
        "⬇ Download Augmented Images (ZIP)",
        zip_buffer.getvalue(),
        "augmented_images.zip",
        "application/zip"
    )
