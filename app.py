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
MODEL_PATH = "models/resnet50_deep_finetuning.pth"

# ---------------- MODEL ----------------
def load_model():
    model = models.resnet50(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 4)

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
        target_layer.register_full_backward_hook(self.backward_hook)

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
    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    return cv2.addWeighted(img, 0.6, heatmap, 0.4, 0)

# ---------------- UI ----------------
st.set_page_config(page_title="Brain Tumor AI", layout="wide")

st.title("🧠 Brain Tumor AI System")
st.markdown("📤 Upload an MRI image to get prediction, confidence scores, and Grad-CAM explanation.")

uploaded_file = st.file_uploader("📁 Upload MRI Image", type=["jpg", "png", "jpeg"])

if uploaded_file:

    image = Image.open(uploaded_file).convert("RGB")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("🖼️ Original MRI Image")
        st.image(image, width=300)

    x = transform(image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        output = model(x)
        probs = torch.softmax(output, dim=1)[0].cpu().numpy()

    pred = int(np.argmax(probs))

    with col2:
        st.subheader("🔍 Prediction Result")
        st.success(f"Predicted Class: {CLASS_NAMES[pred]}")

        st.subheader("🏆 Top-3 Predictions")
        top3 = np.argsort(probs)[::-1][:3]

        for i in top3:
            st.write(f"👉 {CLASS_NAMES[i]}: {probs[i]:.4f}")

        uncertainty = probs[top3[0]] - probs[top3[1]]

        st.subheader("📊 Uncertainty Score")
        st.info(f"Confidence Gap: {uncertainty:.4f}")

        if uncertainty < 0.2:
            st.warning("⚠️ Model is uncertain because the confidence gap is low.")
        else:
            st.success("✅ Model is confident in this prediction.")

    st.subheader("🔥 Model Explainability (Grad-CAM)")

    target_layer = model.layer4[-1]
    gradcam = GradCAM(model, target_layer)

    cam = gradcam.generate(x, pred)
    result_img = overlay_cam(image, cam)

    st.image(result_img, caption="AI Focus Areas using Grad-CAM", width=350)

else:
    st.info("👆 Please upload an MRI image to start the analysis.")