from dotenv import load_dotenv
import torch
import os
import numpy as np
from PIL import Image
from torch import nn
import torchvision
from pytorch_grad_cam.utils.image import (preprocess_image)
import cv2 as cv
import argparse
import json
from model import init_feature_extractor
 
load_dotenv()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_feature_extractor(run_id=None):
    weights_path = None
    model_name = None

    if run_id is not None:
        weights_path = f"{os.getenv('FEATURE_EXTRACTOR_RUNS')}/{run_id}/model.pt"
        with open(f"{os.getenv('FEATURE_EXTRACTOR_RUNS')}/{run_id}/run.json", 'r') as file:
            data = json.load(file)
            if 'model' in data:
                model_name = data['model']
            else:
                model_name = os.getenv('FEATURE_EXTRACTOR')
    else:
        weights_path = os.getenv('FEATURE_EXTRACTOR_MODEL')
        model_name = os.getenv('FEATURE_EXTRACTOR') 

    feature_extractor = init_feature_extractor(model_name=model_name, weights_path=weights_path, device=device)

    return feature_extractor
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("--id")

    args = parser.parse_args()

    run_id = args.id or None

    feature_extractor = get_feature_extractor(run_id)
    
    # target_layers = [model.layer4]

    test_img_path = "./data/kvasir-instrument/images/ckcu9jucf00083b5ytpqoue72.jpg" 

    rgb_img = cv.imread(test_img_path, 1)[:, :, ::-1]
    rgb_img = np.float32(rgb_img) / 255
    input_tensor = preprocess_image(rgb_img,
                                    mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225]).to(device)
    
    # SEZIONE FEATURE EXTR

    output = feature_extractor(input_tensor)

    x = 1

    # SEZIONE GRADCAM

    # targets = [ClassifierOutputTarget(class_names.index('instrument'))]
    # targets = None

    # output = model(input_tensor)
    # class_ = class_names[torch.argmax(output)]

    # with GradCAM(model=model, target_layers=target_layers) as cam:
    #     cam.batch_size = 1

    #     grayscale_cam = cam(input_tensor=input_tensor,
    #                         targets=targets,
    #                         aug_smooth=True,
    #                         eigen_smooth=True)

    #     grayscale_cam = grayscale_cam[0, :]

    #     cam_image = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)
    #     cam_image = cv.cvtColor(cam_image, cv.COLOR_RGB2BGR)

    # gb_model = GuidedBackpropReLUModel(model=model, device=device)
    # gb = gb_model(input_tensor, target_category=None)

    # cam_mask = cv.merge([grayscale_cam, grayscale_cam, grayscale_cam])
    # cam_gb = deprocess_image(cam_mask * gb)
    # gb = deprocess_image(gb)

    # output_dir = 'test'
    # os.makedirs(output_dir, exist_ok=True)

    # cam_output_path = os.path.join(output_dir, f'cam_image.jpg')
    # gb_output_path = os.path.join(output_dir, f'gb.jpg')
    # cam_gb_output_path = os.path.join(output_dir, f'cam_gb.jpg')

    # cv.imwrite(cam_output_path, cam_image)
    # cv.imwrite(gb_output_path, gb)
    # cv.imwrite(cam_gb_output_path, cam_gb)