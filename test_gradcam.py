# test_gradcam_final.py
import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from src.model import DRModelV2
from src.transforms import get_valid_transforms
from src.gradcam import GradCAM, overlay_cam
from src.utils import load_state_dict_compatible

MODEL_PATH = "outputs/checkpoints/best.pth"
IMG_PATH = "data/train_images/0104b032c141.png"   # <- change to an actual file in data/train_images
IMG_SIZE = 512

device = "cuda" if torch.cuda.is_available() else "cpu"

print("Loading model...")
model = DRModelV2().to(device)
ckpt = torch.load(MODEL_PATH, map_location=device)
_, skipped = load_state_dict_compatible(model, ckpt["model_state"])
if skipped:
    print(f"Skipped {len(skipped)} incompatible keys from checkpoint.")
model.eval()

if not os.path.exists(IMG_PATH):
    raise FileNotFoundError(f"Image not found: {IMG_PATH}")
print("Loading image:", IMG_PATH)
bgr = cv2.imread(IMG_PATH)
rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

# preprocessing
tf = get_valid_transforms(IMG_SIZE)
tensor = tf(image=rgb)["image"].unsqueeze(0).to(device)

# create GradCAM (hooks registered in constructor)
# Use the final conv layer from timm EfficientNet backbone
if hasattr(model.backbone, "conv_head"):
    target_layer = model.backbone.conv_head
else:
    target_layer = model.backbone.blocks[-1]

print("Using target layer:", target_layer.__class__.__name__)

cam_gen = GradCAM(model, target_layer)

# generate cam (generate runs forward+backward internally)
print("Generating CAM (forward+backward)...")
cam = cam_gen.generate(tensor)

print("CAM shape, min/max:", cam.shape, cam.min(), cam.max())

overlay = overlay_cam(rgb, cam, alpha=0.45)

plt.figure(figsize=(10,5))
plt.subplot(1,2,1)
plt.title("Original")
plt.imshow(rgb); plt.axis("off")
plt.subplot(1,2,2)
plt.title("Grad-CAM Overlay")
plt.imshow(overlay); plt.axis("off")
plt.show()
print("Done.")
