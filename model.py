from torchvision import models
from dotenv import load_dotenv
from torchvision import models
from datetime import datetime
import logging
import os
import torch.nn as nn
from transformers import AutoTokenizer

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

def get_tokenizer(model_name=os.getenv('TOKENIZER')):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    return tokenizer