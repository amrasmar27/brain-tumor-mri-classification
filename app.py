import streamlit as st
import torch
import torch.nn as nn
import torchvision.models as models
from PIL import Image
import numpy as np
from torchvision import transforms
import streamlit as st

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

# ---------------- UI ----------------
st.title("🧠 Brain Tumor Classification App")
st.write("Upload an MRI image and the model will predict the tumor type.")

uploaded_file = st.file_uploader("Upload MRI Image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    st.image(image, caption="Uploaded Image", width=250)

    img = transform(image)
    img = img.unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        output = model(img)
        probs = torch.softmax(output, dim=1)
        pred = torch.argmax(probs, dim=1).item()

    st.subheader("Prediction:")
    st.success(CLASS_NAMES[pred])

    st.subheader("Confidence:")

    for i, p in enumerate(probs[0]):
        st.write(f"{CLASS_NAMES[i]}: {p.item():.4f}")