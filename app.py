import streamlit as st
from PIL import Image
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from skimage import img_as_ubyte
import os
from runpy import run_path

def process_image(uploaded_image, task):
    img = Image.open(uploaded_image).convert('RGB')
    input_ = TF.to_tensor(img).unsqueeze(0).cuda()

    # Load the correct model file
    if task == "Deblurring":
        weights = os.path.join("Deblurring", "pretrained_models", "deblurring.pth")
    elif task == "Denoising":
        weights = os.path.join("Denoising", "pretrained_models", "denoising.pth")
    elif task == "Deraining":
        weights = os.path.join("Deraining", "pretrained_models", "deraining.pth")

    # Load the model definition
    load_file = run_path(os.path.join(task, "MPRNet.py"))
    model = load_file["MPRNet"]().cuda()

    # Load pretrained weights
    checkpoint = torch.load(weights)
    try:
        model.load_state_dict(checkpoint["state_dict"])
    except:
        new_state = {}
        for k, v in checkpoint["state_dict"].items():
            new_state[k.replace("module.", "")] = v
        model.load_state_dict(new_state)

    model.eval()

    # Pad to multiple of 8
    img_multiple_of = 8
    h, w = input_.shape[2], input_.shape[3]
    H = ((h + img_multiple_of) // img_multiple_of) * img_multiple_of
    W = ((w + img_multiple_of) // img_multiple_of) * img_multiple_of
    padh, padw = H - h, W - w
    input_ = F.pad(input_, (0, padw, 0, padh), 'reflect')

    with torch.no_grad():
        restored = model(input_)[0]
    restored = torch.clamp(restored, 0, 1)[:, :, :h, :w]
    restored = restored.permute(0, 2, 3, 1).cpu().numpy()
    restored = img_as_ubyte(restored[0])

    return restored


# Create the Streamlit app
def main():
    st.title("Image Restoration ")
    st.sidebar.header("Select Task")

    # Task selection (Deblurring, Denoising, or Deraining)
    task = st.sidebar.selectbox('Select Task', ['Deblurring', 'Denoising', 'Deraining'])

    # File uploader for input image
    uploaded_image = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])

    if uploaded_image is not None:
        st.image(uploaded_image, caption="Uploaded Image", use_container_width=True)
        st.write("")

        if st.button('Restore'):
            with st.spinner("Processing..."):
                # Process the image
                result_image = process_image(uploaded_image, task)

            # Display the result image
            st.image(result_image, caption="Processed Image", use_container_width=True)
            st.success("Restore completed!")

# Run the app
if __name__ == "__main__":
    main()
