import streamlit as st
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
.footer { text-align:center; font-size:13px; color:gray; }
</style>
""", unsafe_allow_html=True)

# ================= HEADER =================
st.markdown("<h1>🧪 Image Data Augmentation</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center;'>Upload or Capture an image and generate augmented data</p>", unsafe_allow_html=True)
st.markdown("---")

# ================= INPUT METHOD =================
input_method = st.radio(
    "Choose image input method",
    ["Upload from File Manager", "Capture using Camera"]
)

uploaded_file = (
    st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])
    if input_method == "Upload from File Manager"
    else st.camera_input("Capture Image")
)

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
    st.image(image, caption="Original Image", width="stretch")

    img_array = img_to_array(image)
    img_array = np.expand_dims(img_array, axis=0)

    augmented_images = []
    generator = datagen.flow(img_array, batch_size=1)

    for _ in range(num_images):
        batch = next(generator)
        aug_img = Image.fromarray(batch[0].astype("uint8"))
        augmented_images.append(aug_img)

    # ================= PREVIEW =================
    st.markdown("### 🖼 Augmented Image Preview")
    cols = st.columns(4)
    for i, img in enumerate(augmented_images[:8]):
        cols[i % 4].image(img, width="stretch")

    # ================= ZIP DOWNLOAD =================
    zip_buffer = BytesIO()
    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zip_file:
        for i, img in enumerate(augmented_images):
            img_bytes = BytesIO()
            img.save(img_bytes, format="JPEG")
            zip_file.writestr(f"augmented_{i+1}.jpg", img_bytes.getvalue())

    st.download_button(
        "⬇ Download Augmented Images (ZIP)",
        data=zip_buffer.getvalue(),
        file_name="augmented_images.zip",
        mime="application/zip"
    )

# ================= FOOTER =================
st.markdown("---")
st.markdown(
    "<div class='footer'>Designed for educational & medical AI dataset preparation</div>",
    unsafe_allow_html=True
)
