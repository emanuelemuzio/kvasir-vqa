from model import get_tokenizer
from dotenv import load_dotenv
import os
from feature_extractor import init_feature_extractor
import argparse
import cv2
import numpy as np
import torch
from pytorch_grad_cam.utils.image import (preprocess_image)
import json

load_dotenv()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser()
parser.add_argument('--id')

args = parser.parse_args()

run_id = args.id

resnet = None
weights_path = None

if args.id is not None:
    weights_path = f"{os.getenv('KVASIR_GRADCAM_RUNS')}/{args.id}/model.pt"
    with open(f"{os.getenv('KVASIR_GRADCAM_RUNS')}/{args.id}/run.json", 'r') as file:
        data = json.load(file)
        if 'resnet' in data:
            resnet = data['resnet']
        else:
            resnet = os.getenv('RESNET')
else:
    weights_path = os.getenv('KVASIR_GRADCAM_MODEL')
    resnet = os.getenv('RESNET') 

tokenizer = get_tokenizer()
sentence = "Are there any instruments in the image? Check all that are present."
feature_ext = init_feature_extractor(resnet=resnet, weights_path=weights_path, inference=False, device= device)
tokenizer_output = tokenizer(sentence, add_special_tokens=True, return_tensors='pt', padding='max_length', max_length=14, truncation=True)['input_ids'].squeeze(0).detach().cpu().numpy()

test_img_path = "./data/kvasir-instrument/images/ckcu9jucf00083b5ytpqoue72.jpg"

rgb_img = cv2.imread(test_img_path, 1)[:, :, ::-1]
rgb_img = np.float32(rgb_img) / 255
input_tensor = preprocess_image(rgb_img, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]).to(device)

feature_extractor_output = feature_ext(input_tensor).squeeze(0).squeeze(-1).squeeze(1).detach().cpu().numpy()

joint_embedding = np.multiply(tokenizer_output, feature_extractor_output)

x = 1