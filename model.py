from torchvision import models
from torchvision.models import ResNet152_Weights
from dotenv import load_dotenv
from torch import nn
from torchvision import models
from datetime import datetime
import logging

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

def prepare_pretrained_model(num_classes):
    # # Initialize pre-trained model
    pretrained_model = models.resnet152(weights=ResNet152_Weights.DEFAULT)
    pretrained_model.fc = nn.Linear(pretrained_model.fc.in_features, num_classes)

    # # Freeze the parameters of the pre-trained layers
    for param in pretrained_model.parameters():
        param.requires_grad = False

    logging.info('Froze pre trained layers parameters')

    # # Unfreeze the parameters of the last few layers for fine-tuning
    for param in pretrained_model.layer4.parameters():
        param.requires_grad = True

    logging.info('Unfroze last few layers for fine tuning')

    return pretrained_model
