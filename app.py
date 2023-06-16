import os

import torch
#print(torch.version.cuda)

import torch.nn as nn

from torchvision.models import resnet18, mobilenet_v2, squeezenet1_0, shufflenet_v2_x1_0
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize

from PIL import Image

import streamlit as st


# torchvision transforms
MEAN = [0.5, 0.5, 0.5]
STD = [0.5, 0.5, 0.5]

transform = Compose([
    Resize(256),
    CenterCrop(224),
    ToTensor(),
    Normalize(MEAN, STD)
])


# initializing models
device = "cuda" if torch.cuda.is_available() else "cpu"

resnet = resnet18()
resnet.fc = nn.Linear(resnet.fc.in_features, 7)
resnet.load_state_dict(torch.load("models/emotion_detection_model_resnet.pth", map_location=torch.device('cpu')))
resnet.to(device)
resnet.eval()

mobilenet = mobilenet_v2()
mobilenet.classifier[1] = nn.Linear(mobilenet.classifier[1].in_features, 7)
mobilenet.load_state_dict(torch.load("models/emotion_detection_model_mobilenet.pth", map_location=torch.device('cpu')))
mobilenet.to(device)
mobilenet.eval()

squeezenet = squeezenet1_0()
squeezenet.classifier[1] = nn.Conv2d(squeezenet.classifier[1].in_channels, 7, 1, 1)
squeezenet.load_state_dict(torch.load("models/emotion_detection_model_squeezenet.pth", map_location=torch.device('cpu')))
squeezenet.to(device)
squeezenet.eval()

shufflenet = shufflenet_v2_x1_0()
shufflenet.fc = nn.Linear(shufflenet.fc.in_features, 7)
shufflenet.load_state_dict(torch.load("models/emotion_detection_model_shufflenet.pth", map_location=torch.device('cpu')))
shufflenet.to(device)
shufflenet.eval()

UPLOAD_PATH = "static/temp/"
if not os.path.isdir(UPLOAD_PATH):
    os.mkdir(UPLOAD_PATH)

LABEL_MAP = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']


def predict(image_path, model_name="shufflenet"):
    # PIL Image load and predict
    image = Image.open(image_path)
    image = image.convert("RGB")

    image = transform(image)
    image = image.unsqueeze(0).to(device)

    with torch.no_grad():
        if model_name == "resnet":
            outs = resnet(image)
        elif model_name == "shufflenet":
            outs = shufflenet(image)
        elif model_name == "squeezenet":
            outs = squeezenet(image)
        elif model_name == "mobilenet":
            outs = mobilenet(image)

    outs = torch.argmax(outs, dim=-1)[0]
    outs = LABEL_MAP[outs]

    return outs


def main():
    st.title("Emotion Classifier App")

    model_name = st.selectbox("Select model", ("resnet", "shufflenet", "squeezenet", "mobilenet"))
    image_file = st.file_uploader("Upload an image here", type=["jpg", "jpeg", "png"])

    if image_file is not None:
        image = Image.open(image_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)

        image_path = os.path.join(UPLOAD_PATH, image_file.name)
        image.save(image_path)

        pred = predict(image_path, model_name)
        st.header("Prediction")
        st.write("Predicted emotion:", pred)

if __name__ == "__main__":
    main()

