from transformers import BlipForQuestionAnswering, BlipProcessor
from PIL import Image
import requests
import torch
import data.dataset as dataset_gen
import utils

gpu_available = torch.cuda.is_available()
IMGS_PATH = 'data/val2014/COCO_val2014_[IDENTIFIER].jpg'
IDENTIFIER_LEN = 12

def blip_vqa_test(dataset):
    model_name = "Salesforce/blip-vqa-base"
    model = utils.get_model(BlipForQuestionAnswering, model_name)
    processor = utils.get_model(BlipProcessor, model_name)

    for idx in range(len(dataset)):
        identifier = ('0' * (IDENTIFIER_LEN - len(str(dataset[idx]['image_id'])))) + str(dataset[idx]['image_id'])
        img = Image.open(IMGS_PATH.replace("[IDENTIFIER]", identifier))
        question = dataset[idx]['question']
        inputs = None

        inputs = processor(img, question, return_tensors="pt")
        if gpu_available is True:
            inputs = inputs.to("cuda")
            
        output = model.generate(**inputs)
        processed_output = processor.decode(output[0], skip_special_tokens=True)
        test = 1
        break

def test_models():
    dataset = dataset_gen.build_dataset()
    blip_vqa_test(dataset)
    print(1)

test_models()