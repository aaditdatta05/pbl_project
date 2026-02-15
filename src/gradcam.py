# src/gradcam.py
import torch
import torch.nn.functional as F
import numpy as np
import cv2

class GradCAM:
    """
    Robust Grad-CAM for regression/classification models.
    Usage:
        cam_gen = GradCAM(model, target_layer)
        cam = cam_gen.generate(input_tensor)   # input_tensor shape (1,C,H,W)
    """
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer

        # storage for hooks
        self.gradients = None
        self.activations = None

        # register hooks
        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, inp, out):
            # out shape typically (B, C, H, W)
            self.activations = out.detach()

        def backward_hook(module, grad_in, grad_out):
            # grad_out[0] shape typically (B, C, H, W)
            self.gradients = grad_out[0].detach()

        # forward hook
        self.target_layer.register_forward_hook(forward_hook)

        # choose full backward hook when available (avoids missing grad inputs)
        if hasattr(self.target_layer, "register_full_backward_hook"):
            self.target_layer.register_full_backward_hook(backward_hook)
        else:
            # fallback (older torch) — still works but may warn
            self.target_layer.register_backward_hook(backward_hook)

    def generate(self, x, reshape_transform=None):
        """
        Run forward+backward internally, compute CAM.
        x: torch tensor (1, C, H, W) on same device as model
        reshape_transform: optional fn to transform activations (not used here)
        Returns: 2D numpy array (H, W) float32, values in [0,1]
        """
        # Reset
        self.gradients = None
        self.activations = None

        # forward pass
        self.model.zero_grad()
        out = self.model(x)    # out shape for your DRModel: (1,)

        # get scalar score to backprop:
        # If out has >1 elements, pick first element of batch (out[0])
        if out.numel() == 1:
            score = out
        else:
            # assume batch dimension first; pick first example score
            score = out.view(-1)[0]

        # backward on the chosen score
        score.backward(retain_graph=False)

        # validate hooks captured values
        if self.gradients is None or self.activations is None:
            raise RuntimeError("GradCAM: no gradients or activations captured. Check target_layer.")

        grads = self.gradients[0] if self.gradients.ndim == 4 else self.gradients  # (C,H,W) or (B,C,H,W)
        acts = self.activations[0] if self.activations.ndim == 4 else self.activations

        # if there is still a batch dim, ensure it's removed
        while grads.ndim > 3:
            grads = grads.squeeze(0)
        while acts.ndim > 3:
            acts = acts.squeeze(0)

        # compute weights: global-average-pool over spatial dims
        weights = grads.mean(dim=(1, 2), keepdim=True)   # (C,1,1)
        cam = (weights * acts).sum(dim=0)                # (H,W)

        # ReLU and normalize
        cam = F.relu(cam)
        cam_np = cam.detach().cpu().numpy().astype("float32")

        # handle degenerate case (constant map)
        if cam_np.max() == cam_np.min():
            cam_np = np.zeros_like(cam_np, dtype="float32")
        else:
            cam_np = (cam_np - cam_np.min()) / (cam_np.max() - cam_np.min())

        # ensure 2D
        while cam_np.ndim > 2:
            cam_np = cam_np.squeeze()

        if cam_np.ndim != 2:
            raise RuntimeError(f"GradCAM produced unexpected shape: {cam_np.shape}")

        return cam_np


def overlay_cam(image_rgb, cam, alpha=0.45):
    """
    image_rgb: numpy uint8 RGB image (H,W,3)
    cam: 2D numpy float32 in [0,1]
    returns: overlay RGB uint8 (H,W,3)
    """
    if cam.ndim != 2:
        cam = cam.squeeze()
    H, W, _ = image_rgb.shape

    # Convert cam to uint8 single-channel
    cam_255 = np.clip(cam * 255.0, 0, 255).astype("uint8")

    # Resize to image size
    cam_resized = cv2.resize(cam_255, (W, H), interpolation=cv2.INTER_LINEAR)
    cam_resized = np.ascontiguousarray(cam_resized)

    # apply color map (expects uint8 single channel)
    heatmap = cv2.applyColorMap(cam_resized, cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

    # ensure types
    base = np.ascontiguousarray(image_rgb).astype("uint8")
    heatmap = np.ascontiguousarray(heatmap).astype("uint8")

    overlay = cv2.addWeighted(base, 1 - alpha, heatmap, alpha, 0)
    return overlay.astype("uint8")
