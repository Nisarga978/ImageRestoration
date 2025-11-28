import streamlit as st
from PIL import Image
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from skimage import img_as_ubyte
import os
from runpy import run_path

# Get absolute path to this directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def process_image(uploaded_image, task):
    # Open image
    img = Image.open(uploaded_image).convert('RGB')
    input_ = TF.to_tensor(img).unsqueeze(0)

    # Move to GPU if available
    if torch.cuda.is_available():
        input_ = input_.cuda()

    # -------------------------
    # Correct model weight paths
    # -------------------------
    if task == "Deblurring":
        weights = os.path.join(BASE_DIR, "Deblurring", "pretrained_models", "deblurring.pth")
        model_path = os.path.join(BASE_DIR, "Deblurring", "MPRNet.py")

    elif task == "Denoising":
        weights = os.path.join(BASE_DIR, "Denoising", "pretrained_models", "denoising.pth")
        model_path = os.path.join(BASE_DIR, "Denoising", "MPRNet.py")

    elif task == "Deraining":
        weights = os.path.join(BASE_DIR, "Deraining", "pretrained_models", "deraining.pth")
        model_path = os.path.join(BASE_DIR, "Deraining", "MPRNet.py")

    # Debug (optional)
    print("Loading model from:", model_path)
    print("Loading weights from:", weights)
    print("File exists:", os.path.exists(weights))

    # Load model architecture
    load_file = run_path(model_path)
    model = load_file["MPRNet"]()

    # Move model to GPU
    if torch.cuda.is_available():
        model = model.cuda()

    # Load pretrained weights
    checkpoint = torch.load(weights, map_location="cuda" if torch.cuda.is_available() else "cpu")

    try:
        model.load_state_dict(checkpoint["state_dict"])
    except:
        # Fix for "module." prefix
        new_state = {}
        for k, v in checkpoint["state_dict"].items():
            new_key = k.replace("module.", "")
            new_state[new_key] = v
        model.load_state_dict(new_state)

    model.eval()

    # -------------------------
    # Pad image to multiple of 8
    # -------------------------
    img_multiple_of = 8
    h, w = input_.shape[2], input_.shape[3]
    H = ((h + img_multiple_of) // img_multiple_of) * img_multiple_of
    W = ((w + img_multiple_of) // img_multiple_of) * img_multiple_of
    padh, padw = H - h, W - w

    input_ = F.pad(input_, (0, padw, 0, padh), 'reflect')

    # -------------------------
    # Run inference
    # -------------------------
    with torch.no_grad():
        restored = model(input_)[0]

    restored = torch.clamp(restored, 0, 1)
    restored = restored[:, :, :h, :w]  # Remove padding
    restored = restored.permute(0, 2, 3, 1).cpu().numpy()
    restored = img_as_ubyte(restored[0])

    return restored


# ------------------------------------------------
# Streamlit UI
# ------------------------------------------------
def main():
    st.title("Image Restoration")
    st.sidebar.header("Select Task")

    task = st.sidebar.selectbox('Select Task', ['Deblurring', 'Denoising', 'Deraining'])

    uploaded_image = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])

    if uploaded_image is not None:
        st.image(uploaded_image, caption="Uploaded Image", use_container_width=True)

        if st.button('Restore'):
            with st.spinner("Processing..."):
                result_image = process_image(uploaded_image, task)

            st.image(result_image, caption="Processed Image", use_container_width=True)
            st.success("Restore completed!")


if __name__ == "__main__":
    main()
