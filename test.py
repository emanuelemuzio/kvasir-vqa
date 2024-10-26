from model import embed_question
from dotenv import load_dotenv
import os
from feature_extractor import get_feature_extractor
import argparse
import cv2
import numpy as np
import torch
from pytorch_grad_cam.utils.image import (preprocess_image)
import json
from pytorch_grad_cam.utils.image import (
    show_cam_on_image, deprocess_image, preprocess_image
)
import cv2 as cv
from pytorch_grad_cam import GuidedBackpropReLUModel
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam import GradCAM 
from pytorch_grad_cam.utils.image import show_cam_on_image
from dataset import feature_extractor_class_names 

load_dotenv()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = 'cpu'

parser = argparse.ArgumentParser()
parser.add_argument('--id')

args = parser.parse_args()

run_id = args.id

feature_extractor = get_feature_extractor(run_id=run_id)

class_names = feature_extractor_class_names()
num_classes = len(class_names)
 
sentence = "Are there any instruments in the image? Check all that are present."

test_img_path = "./data/kvasir-instrument/images/ckcu9jucf00083b5ytpqoue72.jpg"
rgb_img = cv2.imread(test_img_path, 1)[:, :, ::-1]
rgb_img = np.float32(rgb_img) / 255
input_tensor = preprocess_image(rgb_img, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]).to(device)

feature_extractor_output = feature_extractor(input_tensor).squeeze()

embedding = embed_question(sentence, device=device)

x = 1

# joint_embedding = np.multiply(tokenizer_output, feature_extractor_output)

# SEZIONE GRADCAM

# targets = [ClassifierOutputTarget(class_names.index('instrument'))]
# targets = None

# target_layers = [model.layer4]

# output = model(input_tensor)
# class_ = class_names[torch.argmax(output)]

# with GradCAM(model=model, target_layers=target_layers) as cam:
#     cam.batch_size = 1

#     grayscale_cam = cam(input_tensor=input_tensor,
#                             targets=targets,
#                             aug_smooth=True,
#                             eigen_smooth=True)

#     grayscale_cam = grayscale_cam[0, :]

#     cam_image = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)
#     cam_image = cv.cvtColor(cam_image, cv.COLOR_RGB2BGR)

# gb_model = GuidedBackpropReLUModel(model=model, device=device)
# gb = gb_model(input_tensor, target_category=None)

# cam_mask = cv2.merge([grayscale_cam, grayscale_cam, grayscale_cam])
# cam_gb = deprocess_image(cam_mask * gb)
# gb = deprocess_image(gb)

# output_dir = 'test'
# os.makedirs(output_dir, exist_ok=True)

# cam_output_path = os.path.join(output_dir, f'cam_image.jpg')
# gb_output_path = os.path.join(output_dir, f'gb.jpg')
# cam_gb_output_path = os.path.join(output_dir, f'cam_gb.jpg')

# cv2.imwrite(cam_output_path, cam_image)
# cv2.imwrite(gb_output_path, gb)
# cv2.imwrite(cam_gb_output_path, cam_gb)