from transformers import AutoModelForImageClassification, AutoFeatureExtractor
from PIL import Image
import requests
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.ablation_layer import AblationLayerVit
from pytorch_grad_cam import GuidedBackpropReLUModel
from pytorch_grad_cam.utils.image import show_cam_on_image, preprocess_image
import cv2

# this is an image from "polyps" class
url = 'https://github.com/mmuratarat/turkish/blob/master/_posts/images/example_polyps_image.jpg?raw=true'
image = Image.open(requests.get(url, stream=True).raw)

model = AutoModelForImageClassification.from_pretrained("mmuratarat/kvasir-v2-classifier")
feature_extractor = AutoFeatureExtractor.from_pretrained("mmuratarat/kvasir-v2-classifier")
inputs = feature_extractor(image, return_tensors="pt")
input_tensor = inputs['pixel_values']

id2label = {'0': 'dyed-lifted-polyps', 
            '1': 'dyed-resection-margins', 
            '2': 'esophagitis', 
            '3': 'normal-cecum', 
            '4': 'normal-pylorus', 
            '5': 'normal-z-line', 
            '6': 'polyps', 
            '7': 'ulcerative-colitis'}

# logits = model(**inputs).logits
# predicted_label = logits.argmax(-1).item()
# predicted_class = id2label[str(predicted_label)]

target_layers = [model.vit.layernorm]

cam = GradCAM(model=model, target_layers=target_layers, reshape_transform=None)

targets = None

grayscale_cam = cam(input_tensor=input_tensor, targets=targets)

# Here grayscale_cam has only one image in the batch
grayscale_cam = grayscale_cam[0, :]

cam_image = show_cam_on_image(input_tensor, grayscale_cam)
cv2.imwrite(f'test/_cam.jpg', cam_image)

# print(predicted_class)