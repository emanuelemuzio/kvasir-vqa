from transformers import BlipForQuestionAnswering, BlipProcessor, ViltProcessor, ViltForQuestionAnswering, AutoProcessor, AutoModelForCausalLM
from PIL import Image
import requests
import torch
import data.dataset as dataset_gen
from data.dataset import load_json, ANNOTATIONS_PATH
import utils
import os
import csv
import argparse

gpu_available = torch.cuda.is_available()
IDENTIFIER_LEN = 12 

"""

Function that is generically used for determining the accuracy of VQA answers.
Will be the minimum value between the # of humans that provided the same answer and 1

"""

def accuracyVQA(my_answer: str, answers: list) -> float:
    accuracy = 0
    for answer in answers:
        if answer['answer'].lower() == my_answer.lower():
            accuracy = accuracy + 1
    return min(accuracy / len(answers), 1)

"""

https://huggingface.co/Salesforce/blip-vqa-base

"""

def blip_vqa_test(dataset) -> float:
    model_name = "Salesforce/blip-vqa-base"
    model = utils.get_model(BlipForQuestionAnswering, model_name)
    processor = utils.get_model(BlipProcessor, model_name)
    annotations = load_json(ANNOTATIONS_PATH)['annotations']
    accuracy = 0

    for idx in range(len(dataset)):
        identifier = ('0' * (IDENTIFIER_LEN - len(str(dataset[idx]['image_id'])))) + str(dataset[idx]['image_id'])
        img = Image.open(IMGS_PATH.replace("[IDENTIFIER]", identifier))
        question = dataset[idx]['question']
        inputs = None
        answers = annotations[idx]['answers']

        inputs = processor(img, question, return_tensors="pt")
        if gpu_available is True:
            inputs = inputs.to("cuda")
            model = model.to("cuda")
            
        output = model.generate(**inputs)
        processed_output = processor.decode(output[0], skip_special_tokens=True)
        accuracy = accuracy + accuracyVQA(processed_output, answers)

        del inputs
    
    del model
    
    return accuracy / len(annotations)

"""

https://huggingface.co/dandelin/vilt-b32-finetuned-vqa

"""

def vilt_vqa_test(dataset) -> float:
    model_name = "dandelin/vilt-b32-finetuned-vqa"
    model = utils.get_model(ViltForQuestionAnswering, model_name)
    processor = utils.get_model(ViltProcessor, model_name)
    annotations = load_json(ANNOTATIONS_PATH)['annotations']
    accuracy = 0

    for idx in range(len(dataset)):
        identifier = ('0' * (IDENTIFIER_LEN - len(str(dataset[idx]['image_id'])))) + str(dataset[idx]['image_id'])
        img = Image.open(IMGS_PATH.replace("[IDENTIFIER]", identifier))
        question = dataset[idx]['question']
        inputs = None
        answers = annotations[idx]['answers']

        inputs = processor(img, question, return_tensors="pt")
        if gpu_available is True:
            inputs = inputs.to("cuda")
            model = model.to("cuda")
            
        outputs = model(**inputs)
        logits = outputs.logits
        output_idx = logits.argmax(-1).item()
        processed_output = model.config.id2label[output_idx]
        accuracy = accuracy + accuracyVQA(processed_output, answers)

        del inputs
    
    del model
    
    return accuracy / len(annotations)


"""

https://huggingface.co/microsoft/git-base-vqav2

"""

def git_base_vqa_test(dataset) -> float:
    model_name = "microsoft/git-base-vqav2"
    model = utils.get_model(AutoModelForCausalLM, model_name)
    processor = utils.get_model(AutoProcessor, model_name)
    annotations = load_json(ANNOTATIONS_PATH)['annotations']
    accuracy = 0

    for idx in range(len(dataset)):
        identifier = ('0' * (IDENTIFIER_LEN - len(str(dataset[idx]['image_id'])))) + str(dataset[idx]['image_id'])
        img = Image.open(IMGS_PATH.replace("[IDENTIFIER]", identifier)).convert("RGB")
        question = dataset[idx]['question']
        answers = annotations[idx]['answers']
        pixel_values = processor(images=img, return_tensors="pt").pixel_values

        input_ids = processor(text=question, add_special_tokens=False).input_ids
        input_ids = [processor.tokenizer.cls_token_id] + input_ids
        input_ids = torch.tensor(input_ids).unsqueeze(0)

        if gpu_available is True:
            input_ids = input_ids.to("cuda")
            pixel_values = pixel_values.to("cuda")
            model = model.to("cuda")

        generated_ids = model.generate(pixel_values=pixel_values, input_ids=input_ids, max_length=50)
        processed_output = processor.batch_decode(generated_ids, skip_special_tokens=True) 

        processed_output = processed_output[0].replace(question.lower(), '').strip()

        accuracy = accuracy + accuracyVQA(processed_output, answers)

        del inputs
    
    del model
    
    return accuracy / len(annotations)

"""

Generate the file tests.csv, containing the accuracy of all the models being tested
The models are hardcoded and since each one slightly differs from the others, has a dedicated function
for inference.
TODO: Differentiate the test for abstract and real scenes, ATM only real scenes are considered

"""
MODELS = [
    
    ''
    ''
]
def test_models():
    if not os.path.exists('blip-vqa-base_tests.csv'):
        dataset = dataset_gen.build_dataset()
        blip_vqa_accuracy = blip_vqa_test(dataset)
        with open('blip-vqa-base_tests.csv', 'w', newline='') as file:
            writer = csv.writer(file)
            field = ["model", "accuracy","scene"]
            writer.writerow(field)
            writer.writerow(["Salesforce/blip-vqa-base", blip_vqa_accuracy, "COCO"])
    
    if not os.path.exists('vilt-b32-finetuned-vqa_tests.csv'):
        dataset = dataset_gen.build_dataset()
        vilt_vqa_accuracy = vilt_vqa_test(dataset)
        with open('vilt-b32-finetuned-vqa_tests.csv', 'w', newline='') as file:
            writer = csv.writer(file)
            field = ["model", "accuracy","scene"]
            writer.writerow(field)
            writer.writerow(["dandelin/vilt-b32-finetuned-vqa", vilt_vqa_accuracy, "COCO"])

    if not os.path.exists('git-base-vqav2_tests.csv'):
        dataset = dataset_gen.build_dataset()
        git_base_vqa_accuracy = git_base_vqa_test(dataset)
        with open("git-base-vqav2_tests.csv", 'w', newline='') as file:
            writer = csv.writer(file)
            field = ["model", "accuracy","scene"]
            
            writer.writerow(field)
            writer.writerow(["microsoft/git-base-vqav2", git_base_vqa_accuracy, "COCO"])

def main():
    parser = argparse.ArgumentParser(description ='Process some integers.')
    parser.add_argument('integers', metavar ='N', 
                    type = int, nargs ='+',
                    help ='an integer for the accumulator')
 
    parser.add_argument(dest ='accumulate', 
                        action ='store_const',
                        const = sum, 
                        help ='sum the integers')
    
    args = parser.parse_args()
    test_models()

if __name__ == '__main__':
    main()