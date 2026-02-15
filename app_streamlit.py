import streamlit as st
import cv2
import numpy as np
import torch
from PIL import Image

from src.model import DRModelV2
from src.transforms import get_valid_transforms
from src.gradcam import GradCAM, overlay_cam
from src.utils import load_state_dict_compatible

st.title("Diabetic Retinopathy Detection")

device = "cuda" if torch.cuda.is_available() else "cpu"

@st.cache_resource
def load_model():
    model = DRModelV2().to(device)
    ckpt = torch.load("outputs/checkpoints/dr.pth", map_location=device)
    _, skipped = load_state_dict_compatible(model, ckpt["model_state"])
    if skipped:
        st.info(f"Skipped {len(skipped)} incompatible keys from checkpoint.")
    model.eval()
    return model

model = load_model()

uploaded = st.file_uploader("Upload fundus image", type=["png","jpg","jpeg"])

if uploaded:
    file_bytes = np.asarray(bytearray(uploaded.read()), dtype=np.uint8)
    img_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    st.image(img_rgb, caption="Original", use_container_width=True)

    tf = get_valid_transforms(512)
    sample = tf(image=img_rgb)["image"].unsqueeze(0).to(device)

    model.zero_grad()
    grades, probs = model.predict_ordinal(sample)
    pred = int(grades.item())
    raw_pred = probs.squeeze(0).detach().cpu().numpy().tolist()

    st.write(f"Predicted DR Grade: {pred}")
    st.write(f"Cumulative probs: {raw_pred}")

    if hasattr(model.backbone, "conv_head"):
        target_layer = model.backbone.conv_head
    else:
        target_layer = model.backbone.blocks[-1]

    cam_gen = GradCAM(model, target_layer)
    cam = cam_gen.generate(sample)
    overlay = overlay_cam(img_rgb, cam)
    st.image(overlay, caption="Grad-CAM Heatmap", use_container_width=True)