import streamlit as st
import zipfile
import random
from PIL import Image, ImageEnhance
from io import BytesIO

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

num_images = st.slider("Number of augmented images", 1, 50, 20)

# ================= AUGMENT FUNCTION =================
def augment_image(img):
    img = img.rotate(random.randint(-20, 20))
    if random.random() > 0.5:
        img = img.transpose(Image.FLIP_LEFT_RIGHT)

    img = ImageEnhance.Brightness(img).enhance(random.uniform(0.8, 1.2))
    img = ImageEnhance.Contrast(img).enhance(random.uniform(0.8, 1.2))
    return img

# ================= BUTTON =================
if st.button("Generate Augmented Images") and uploaded_file:
    try:
        image = Image.open(uploaded_file).convert("RGB")
        image.thumbnail((512, 512))

        augmented_images = [augment_image(image.copy()) for _ in range(num_images)]
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

    except Exception:
        st.error("Invalid image file. Please try another image.")

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
