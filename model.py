from torchvision import models
from dotenv import load_dotenv
from torchvision import models
from datetime import datetime
import logging
import torch
from torchvision.transforms import v2
from dataset import prepare_data, kvasir_gradcam_class_names

now = datetime.now()
now = now.strftime("%Y-%m-%d")

logging.basicConfig(
    filename=f"logs/{now}.log",
    encoding="utf-8",
    filemode="a",
    format="{asctime} - {levelname} - {message}",
    style="{",
    datefmt="%Y-%m-%d %H:%M",
    force=True,
    level=logging.INFO
)

load_dotenv()

def prepare_pretrained_model(resnet='152', num_classes=0, freeze_layers=False, inference=False):
    pretrained_model = None

    if resnet == '152':
        pretrained_model = models.resnet152()
    elif resnet == '101':
        pretrained_model = models.resnet101()
    elif resnet == '50':
        pretrained_model = models.resnet50()
    
    pretrained_model.fc = nn.Linear(pretrained_model.fc.in_features, num_classes)

    if not inference:

        for param in pretrained_model.parameters():
            param.requires_grad = not freeze_layers

        for param in pretrained_model.layer4.parameters():
            param.requires_grad = True


    return pretrained_model

def transform():

    transform = v2.Compose([
        v2.Resize((224,224)),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    return transform 