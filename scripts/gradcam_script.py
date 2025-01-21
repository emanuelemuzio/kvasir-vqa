

import sys
sys.path.append('src')

import warnings
warnings.filterwarnings('ignore')
 
import argparse
import os
import cv2
import numpy as np
import torch 
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image, preprocess_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
import os
from dotenv import load_dotenv
from feature_extractor.model import get_model
from common.util import ROOT

load_dotenv()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = 'cpu'
 
if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    experiment = 'instrument'

    polyp_path = f"{ROOT}/{os.getenv('KVASIR_VQA_DATA')}/cl8k2u1pw1e6j0832cvtlam9l.jpg"
    instrument_path = f"{ROOT}/{os.getenv('KVASIR_VQA_DATA')}/clb0lbwz7doug086u3av4f1xw.jpg" 
    ulcerative_colitis_path = f"{ROOT}/{os.getenv('KVASIR_VQA_DATA')}/cla820glrs4v3071u41toe6lv.jpg"
    
    image_path = instrument_path
    
    resnet50_weights_path = f"{ROOT}/{os.getenv('FEATURE_EXTRACTOR_RUNS')}/23122024203310/model.pt"
    resnet101_weights_path = f"{ROOT}/{os.getenv('FEATURE_EXTRACTOR_RUNS')}/23122024235333/model.pt"
    resnet152_weights_path = f"{ROOT}/{os.getenv('FEATURE_EXTRACTOR_RUNS')}/24122024135401/model.pt"
    vgg_weights_path = f"{ROOT}/{os.getenv('FEATURE_EXTRACTOR_RUNS')}/25122024000119/model.pt"
    vit_weights_path = f"{ROOT}/{os.getenv('FEATURE_EXTRACTOR_RUNS')}/25122024050535/model.pt" 
    
    feature_extractor_weights_path = resnet152_weights_path
    
    model = get_model('resnet152', 24, '2').to(device)
    weights = torch.load(feature_extractor_weights_path)
    model.load_state_dict(weights)
    
    rgb_img = cv2.imread(image_path, 1)[:, :, ::-1]
    rgb_img = np.float32(rgb_img) / 255
    input_tensor = preprocess_image(rgb_img,
                                    mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225]).to(device)
    
    # Resnet50: model.layer4[-1]
    # VGG, densenet161 and mobilenet: model.features[-1]
    # ViT: model.blocks[-1].norm1
    
    target_layers = [model.layer4[-1]]
    
    # Create an input tensor image for your model..
    # Note: input_tensor can be a batch tensor with several images!

    # We have to specify the target we want to generate the CAM for.
    # targets = [ClassifierOutputTarget(1)]
    targets = None

    with GradCAM(model=model, target_layers=target_layers) as cam:

        # AblationCAM and ScoreCAM have batched implementations.
        # You can override the internal batch size for faster computation.
        cam.batch_size = 1
        grayscale_cam = cam(input_tensor=input_tensor, targets=targets)

        grayscale_cam = grayscale_cam[0, :]

        cam_image = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)
        cam_image = cv2.cvtColor(cam_image, cv2.COLOR_RGB2BGR)

    cam_mask = cv2.merge([grayscale_cam, grayscale_cam, grayscale_cam])

    cam_output_path = f"{ROOT}/experiments/{experiment}_gradcam.jpg"

    cv2.imwrite(cam_output_path, cam_image)