import os
from transformers import AutoModelForImageClassification, AutoFeatureExtractor
from PIL import Image
import requests
from dotenv import load_dotenv
from matplotlib import pyplot as plt
from torchvision.transforms import functional as F

load_dotenv()

feature_extractor_name = os.getenv("FEATURE_EXTRACTOR")

# Citazioni
## Classificatore KVASIR del bro su github
## ViT 
## KVASIR-VQA
##  

id2label = {'0': 'dyed-lifted-polyps', 
            '1': 'dyed-resection-margins', 
            '2': 'esophagitis', 
            '3': 'normal-cecum', 
            '4': 'normal-pylorus', 
            '5': 'normal-z-line', 
            '6': 'polyps', 
            '7': 'ulcerative-colitis'}

def get_feature_extractor():
    feature_extractor = AutoFeatureExtractor.from_pretrained(feature_extractor_name)
    return feature_extractor

def get_classifier():
    model = AutoModelForImageClassification.from_pretrained(feature_extractor_name)
    return model

def extract_features(path: str):
    feature_extractor = get_feature_extractor()
    classifier = get_classifier()
    image = Image.open(path)    
    inputs = feature_extractor(image, return_tensors="pt")
    
    feature = inputs.data['pixel_values'][0]
    im = F.to_pil_image(feature)
    # im.show()
    
    logits = classifier(**inputs).logits
    predicted_label = logits.argmax(-1).item()
    predicted_class = id2label[str(predicted_label)]
    
    print(predicted_class)

if __name__ == '__main__':
    # test = ["data\img\clb0lbx06dq4g086uc8pt0brb.jpg","data\img\cl8k2u1pn1dxf083248iz2qqy.jpg"]
    # for t in test:
    #     extract_features(t)
    print(1)
    