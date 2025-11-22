import streamlit as st
from PIL import Image
import torch
import torch.nn as nn
from torchvision import transforms, models
import numpy as np

# ------------------------------
# Настройки
# ------------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
model_path = "best_car_brand_model.pth"

# ------------------------------
# Загрузка модели
# ------------------------------
checkpoint = torch.load(model_path, map_location=device)
num_classes = checkpoint["num_classes"]
classes = checkpoint["classes"]
model_name = checkpoint["model_name"]
IMG_SIZE = checkpoint["img_size"]

def get_model(model_name, num_classes):
    if model_name == "efficientnet":
        model = models.efficientnet_b0(weights=None)
        in_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(in_features, num_classes)
    elif model_name == "resnet":
        model = models.resnet18(weights=None)
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)
    elif model_name == "densenet":
        model = models.densenet121(weights=None)
        in_features = model.classifier.in_features
        model.classifier = nn.Linear(in_features, num_classes)
    elif model_name == "vgg":
        model = models.vgg16(weights=None)
        in_features = model.classifier[6].in_features
        model.classifier[6] = nn.Linear(in_features, num_classes)
    else:
        raise ValueError("Unknown model name")
    return model

model = get_model(model_name, num_classes)
model.load_state_dict(checkpoint["model_state_dict"])
model.to(device)
model.eval()

# ------------------------------
# Функция предсказания
# ------------------------------
CONF_THRESHOLD = 0.40   # 40%

def predict(image: Image.Image):
    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    ])
    img_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(img_tensor)
        probs = torch.softmax(outputs, dim=1)[0]   # shape [num_classes]

    # Top-1
    conf, pred_idx = torch.max(probs, dim=0)
    conf = conf.item()
    pred_class = classes[pred_idx.item()]

    # Top-3
    top3_conf, top3_idx = torch.topk(probs, k=3)
    top3_classes = [(classes[i], float(c)) for i, c in zip(top3_idx, top3_conf)]

    return pred_class, conf, top3_classes


# ------------------------------
# Streamlit UI
# ------------------------------
st.title("Car Brand Classifier 🚗")
st.write("Загрузи фото автомобиля и узнай марку!")

uploaded_file = st.file_uploader("Выбери изображение", type=["jpg","jpeg","png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Загруженное изображение", use_container_width=True)
    st.write("Предсказание...")

    label, confidence, top3 = predict(image)

    if confidence < CONF_THRESHOLD:
        st.error(
            f"❗ Модель **не уверена** в результате (< {CONF_THRESHOLD*100:.0f}%).\n"
            f"Попробуйте другое фото.\n\n"
            f"Возможные варианты (Top-3):"
        )
        for cls, conf in top3:
            st.write(f"- **{cls}** ({conf*100:.1f}%)")
    else:
        st.success(f"Марка: **{label}** (уверенность: {confidence*100:.2f}%)")