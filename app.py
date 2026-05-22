import streamlit as st
import torch
import torch.nn as nn
import torchvision.models as models
import numpy as np
import cv2
from PIL import Image
from torchvision import transforms

# ---------------- CONFIG ----------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CLASS_NAMES = ["glioma", "meningioma", "no_tumor", "pituitary"]

MODEL_PATH = "models/mobilenet_block_unfreeze.pth"

# ---------------- MODEL ----------------
def load_model():
    model = models.mobilenet_v2(weights=None)
    model.classifier[1] = nn.Linear(model.last_channel, 4)

    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()

    return model

model = load_model()

# ---------------- TRANSFORM ----------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# ---------------- GRAD-CAM ----------------
class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.gradients = None
        self.activations = None

        target_layer.register_forward_hook(self.forward_hook)
        target_layer.register_backward_hook(self.backward_hook)

    def forward_hook(self, module, input, output):
        self.activations = output

    def backward_hook(self, module, grad_in, grad_out):
        self.gradients = grad_out[0]

    def generate(self, x, class_idx):
        self.model.zero_grad()

        output = self.model(x)

        loss = output[0, class_idx]
        loss.backward()

        grads = self.gradients
        acts = self.activations

        weights = torch.mean(grads, dim=(2, 3), keepdim=True)
        cam = torch.sum(weights * acts, dim=1)

        cam = torch.relu(cam).squeeze().detach().cpu().numpy()

        cam = cv2.resize(cam, (224, 224))
        cam = (cam - cam.min()) / (cam.max() + 1e-8)

        return cam


def overlay_cam(img, cam):
    img = np.array(img.resize((224, 224)))

    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_VIRIDIS)
    overlay = cv2.addWeighted(img, 0.6, heatmap, 0.4, 0)

    return overlay

# ---------------- UI ----------------
st.set_page_config(page_title="Brain Tumor AI", layout="centered")

st.title("🧠 Brain Tumor Classification + Explainability (Grad-CAM)")
st.write("Upload MRI image to predict tumor type and visualize AI reasoning.")

uploaded_file = st.file_uploader("Upload MRI Image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    st.image(image, caption="Original Image", width=250)

    # preprocess
    x = transform(image).unsqueeze(0).to(DEVICE)

    # prediction
    with torch.no_grad():
        output = model(x)
        probs = torch.softmax(output, dim=1)
        pred = torch.argmax(probs, dim=1).item()

    st.subheader("🧠 Prediction:")
    st.success(CLASS_NAMES[pred])

    st.subheader("📊 Confidence:")
    for i, p in enumerate(probs[0]):
        st.write(f"{CLASS_NAMES[i]}: {p.item():.4f}")

    # ---------------- GRAD-CAM ----------------
    st.subheader("Grad-CAM Explanation")

    target_layer = model.features[-1]
    gradcam = GradCAM(model, target_layer)

    cam = gradcam.generate(x, pred)

    result_img = overlay_cam(image, cam)

    st.image(result_img, caption="Grad-CAM Heatmap", width=250)