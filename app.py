import streamlit as st
import os
import zipfile
import numpy as np
from PIL import Image
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array
from io import BytesIO

# ================= PAGE CONFIG =================
st.set_page_config(
    page_title="Image Data Augmentation",
    page_icon="🧪",
    layout="centered"
)

# ================= STYLE =================
st.markdown("""
<style>
body { background-color: #f4f6f9; }
h1 { color: #0b5394; font-weight: 700; text-align: center; }
.box {
    padding: 20px;
    background: #ffffff;
    border-radius: 12px;
    box-shadow: 0 4px 10px rgba(0,0,0,0.1);
}
.footer { text-align:center; font-size:13px; color:gray; }
</style>
""", unsafe_allow_html=True)

# ================= HEADER =================
st.markdown("<h1>🧪 Image Data Augmentation</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center;'>Images and Generate Augmented Data</p>", unsafe_allow_html=True)
st.markdown("---")

# ================= INPUT METHOD =================
st.markdown("### 📥 Input Image")
input_method = st.radio(
    "Choose image input method",
    ["Upload from File Manager", "Capture using Camera"]
)

uploaded_file = None

if input_method == "Upload from File Manager":
    uploaded_file = st.file_uploader(
        "Upload Image",
        type=["jpg", "jpeg", "png"]
    )
else:
    uploaded_file = st.camera_input("Capture Image")

# ================= AUGMENTATION SETTINGS =================
st.markdown("### ⚙️ Augmentation Settings")

num_images = st.slider("Number of augmented images", 1, 50, 20)

datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode="nearest"
)

# ================= PROCESS =================
if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Original Image", use_container_width=True)

    img_array = img_to_array(image)
    img_array = np.expand_dims(img_array, axis=0)

    augmented_images = []

    for batch in datagen.flow(img_array, batch_size=1):
        aug_img = Image.fromarray(batch[0].astype("uint8"))
        augmented_images.append(aug_img)
        if len(augmented_images) >= num_images:
            break

    # ================= PREVIEW =================
    st.markdown("### 🖼 Augmented Image Preview")
    cols = st.columns(4)
    for i, img in enumerate(augmented_images[:8]):
        cols[i % 4].image(img, use_container_width=True)

    # ================= ZIP DOWNLOAD =================
    zip_buffer = BytesIO()
    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zip_file:
        for i, img in enumerate(augmented_images):
            img_bytes = BytesIO()
            img.save(img_bytes, format="JPEG")
            zip_file.writestr(f"augmented_{i+1}.jpg", img_bytes.getvalue())

    st.download_button(
        label="⬇ Download Augmented Images (ZIP)",
        data=zip_buffer.getvalue(),
        file_name="augmented_images.zip",
        mime="application/zip"
    )

# ================= FOOTER =================
st.markdown("---")
st.markdown(
    "<div class='footer'>⚕️ Designed for educational & medical AI dataset preparation</div>",
    unsafe_allow_html=True
)
