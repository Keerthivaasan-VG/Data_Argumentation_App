import streamlit as st
import zipfile
import random
from PIL import Image, ImageEnhance
from io import BytesIO

st.set_page_config(
    page_title="Image Data Augmentation",
    page_icon="🧪",
    layout="centered"
)

st.markdown("<h1 style='text-align:center;'>🧪 Image Data Augmentation</h1>", unsafe_allow_html=True)
st.markdown("---")

# ---------------- STATE ----------------
if "images" not in st.session_state:
    st.session_state.images = None
if "zipdata" not in st.session_state:
    st.session_state.zipdata = None

# ---------------- INPUT ----------------
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

# ---------------- AUGMENT ----------------
def augment(img):
    img = img.rotate(random.randint(-25, 25))
    if random.random() > 0.5:
        img = img.transpose(Image.FLIP_LEFT_RIGHT)
    img = ImageEnhance.Brightness(img).enhance(random.uniform(0.8, 1.2))
    img = ImageEnhance.Contrast(img).enhance(random.uniform(0.8, 1.2))
    return img

# ---------------- BUTTON ----------------
if st.button("Generate Augmented Images") and uploaded_file:
    try:
        base_img = Image.open(uploaded_file).convert("RGB")
        base_img.thumbnail((512, 512))

        images = [augment(base_img.copy()) for _ in range(num_images)]
        st.session_state.images = images

        zip_buffer = BytesIO()
        with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as z:
            for i, im in enumerate(images):
                buf = BytesIO()
                im.save(buf, format="JPEG")
                z.writestr(f"augmented_{i+1}.jpg", buf.getvalue())

        zip_buffer.seek(0)
        st.session_state.zipdata = zip_buffer.getvalue()
        st.success("Augmented images generated successfully!")

    except Exception:
        st.error("Please upload a valid image.")

# ---------------- OUTPUT ----------------
if st.session_state.images:
    st.markdown("### 🖼 Preview")
    cols = st.columns(4)
    for i, im in enumerate(st.session_state.images[:8]):
        cols[i % 4].image(im, width="stretch")

    st.download_button(
        "⬇ Download Augmented Images (ZIP)",
        st.session_state.zipdata,
        "augmented_images.zip",
        "application/zip"
    )
