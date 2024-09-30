from torchvision.models import resnet50 as resnet
from dotenv import load_dotenv
import torch
import os
import numpy as np
from PIL import Image
import torchvision
from torchvision.models import ResNet50_Weights
from pytorch_grad_cam import GradCAM 
from pytorch_grad_cam.utils.image import show_cam_on_image
from kvasir_gradcam import prepare_data
from torch import nn
from pytorch_grad_cam.utils.image import (
    show_cam_on_image, deprocess_image, preprocess_image
)
import cv2 as cv
from pytorch_grad_cam import GuidedBackpropReLUModel
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

load_dotenv()

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = 'cpu'

def get_img(path: str):
    rgb_img = Image.open(path).convert('RGB')
    rgb_img = (rgb_img - np.min(rgb_img)) / (np.max(rgb_img) - np.min(rgb_img))
    input_tensor = torchvision.transforms.functional.to_tensor(rgb_img).unsqueeze(0).float()
    return rgb_img, input_tensor

if __name__ == '__main__':
    dataset = prepare_data()
    class_names = dataset.label.unique()
    num_classes = len(class_names)

    model = resnet(pretrained=True).to(device).eval()
    model.fc = nn.Linear(model.fc.in_features, num_classes) 

    model.load_state_dict(torch.load(os.getenv('KVASIR_GRADCAM_MODEL'), weights_only=True))
    target_layers = [model.layer4]

    test_img_path = "./data/hyper-kvasir/labeled-images/lower-gi-tract/pathological-findings/polyps/0a6957d3-6e48-4756-aa0e-6e185e417618.jpg" 

    rgb_img = cv.imread(test_img_path, 1)[:, :, ::-1]
    rgb_img = np.float32(rgb_img) / 255
    input_tensor = preprocess_image(rgb_img,
                                    mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225]).to(device)
    
    targets = None

    with GradCAM(model=model, target_layers=target_layers) as cam:
        cam.batch_size = 32

        grayscale_cam = cam(input_tensor=input_tensor,
                            targets=targets,
                            aug_smooth=True,
                            eigen_smooth=True)

        grayscale_cam = grayscale_cam[0, :]

        cam_image = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)
        cam_image = cv.cvtColor(cam_image, cv.COLOR_RGB2BGR)

    gb_model = GuidedBackpropReLUModel(model=model, device=device)
    gb = gb_model(input_tensor, target_category=None)

    cam_mask = cv.merge([grayscale_cam, grayscale_cam, grayscale_cam])
    cam_gb = deprocess_image(cam_mask * gb)
    gb = deprocess_image(gb)

    output_dir = 'test'
    os.makedirs(output_dir, exist_ok=True)

    cam_output_path = os.path.join(output_dir, f'grad_cam.jpg')
    gb_output_path = os.path.join(output_dir, f'grad_cam_gb.jpg')
    cam_gb_output_path = os.path.join(output_dir, f'grad_cam_gb.jpg')

    cv.imwrite(cam_output_path, cam_image)
    cv.imwrite(gb_output_path, gb)
    cv.imwrite(cam_gb_output_path, cam_gb)