from transformers import BlipForQuestionAnswering, BlipProcessor, ViltProcessor, ViltForQuestionAnswering, AutoProcessor, AutoModelForCausalLM
from PIL import Image
import torch
import dataset as dataset_gen
from dataset import load_json
import utils
import os
import csv
import argparse
from utils import load_json
from tqdm.auto import tqdm

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

total accuracy = accuracy / N questions, here it will be formatted as an integer up to 100%

"""

def format_accuracy(accuracy, N):
    return ("{:.2f}%".format(round((accuracy / N) * 100, 2)))


"""

https://huggingface.co/Salesforce/blip-vqa-base

"""

def blip_vqa_test(dataset, annotations_path: str, imgs_path: str) -> float:
    model_name = "Salesforce/blip-vqa-base"
    model = utils.get_model(BlipForQuestionAnswering, model_name)
    processor = utils.get_model(BlipProcessor, model_name)
    annotations = load_json(annotations_path)['annotations']
    accuracy = 0

    if gpu_available is True:
        model = model.to("cuda")

    for idx in tqdm(range(len(dataset)), desc='Blip VQA Base Testing'):
        identifier = ('0' * (IDENTIFIER_LEN - len(str(dataset[idx]['image_id'])))) + str(dataset[idx]['image_id'])
        img = Image.open(imgs_path.replace("[ID]", identifier))
        question = dataset[idx]['question']
        inputs = None
        answers = annotations[idx]['answers']

        inputs = processor(img, question, return_tensors="pt")
        if gpu_available is True:
            inputs = inputs.to("cuda") 
            
        output = model.generate(**inputs)
        processed_output = processor.decode(output[0], skip_special_tokens=True)
        accuracy = accuracy + accuracyVQA(processed_output, answers)

        del inputs
    
    del model

    return format_accuracy(accuracy, len(annotations))

"""

https://huggingface.co/dandelin/vilt-b32-finetuned-vqa

"""

def vilt_vqa_test(dataset, annotations_path: str, imgs_path: str) -> float:
    model_name = "dandelin/vilt-b32-finetuned-vqa"
    model = utils.get_model(ViltForQuestionAnswering, model_name)
    processor = utils.get_model(ViltProcessor, model_name)
    annotations = load_json(annotations_path)['annotations']
    accuracy = 0

    if gpu_available is True:
        model = model.to("cuda")

    for idx in tqdm(range(len(dataset)), desc='Vilt 32B VQA Testing'):
        identifier = ('0' * (IDENTIFIER_LEN - len(str(dataset[idx]['image_id'])))) + str(dataset[idx]['image_id'])
        img = Image.open(imgs_path.replace("[ID]", identifier)).convert("RGB")
        question = dataset[idx]['question']
        inputs = None
        answers = annotations[idx]['answers']

        inputs = processor(img, question, return_tensors="pt")
        if gpu_available is True:
            inputs = inputs.to("cuda") 
            
        outputs = model(**inputs)
        logits = outputs.logits
        output_idx = logits.argmax(-1).item()
        processed_output = model.config.id2label[output_idx]
        accuracy = accuracy + accuracyVQA(processed_output, answers)

        del inputs
    
    del model
    
    return format_accuracy(accuracy, len(annotations))


"""

https://huggingface.co/microsoft/git-base-vqav2

"""

def git_base_vqa_test(dataset, annotations_path: str, imgs_path: str) -> float:
    model_name = "microsoft/git-base-vqav2"
    model = utils.get_model(AutoModelForCausalLM, model_name)
    processor = utils.get_model(AutoProcessor, model_name)
    annotations = load_json(annotations_path)['annotations']
    accuracy = 0

    if gpu_available is True:
        model = model.to("cuda")

    for idx in tqdm(range(len(dataset)), desc='Git Base VQA V2 Testing'):
        identifier = ('0' * (IDENTIFIER_LEN - len(str(dataset[idx]['image_id'])))) + str(dataset[idx]['image_id'])
        img = Image.open(imgs_path.replace("[ID]", identifier)).convert("RGB")
        question = dataset[idx]['question']
        answers = annotations[idx]['answers']
        pixel_values = processor(images=img, return_tensors="pt").pixel_values

        input_ids = processor(text=question, add_special_tokens=False).input_ids
        input_ids = [processor.tokenizer.cls_token_id] + input_ids
        input_ids = torch.tensor(input_ids).unsqueeze(0)

        if gpu_available is True:
            input_ids = input_ids.to("cuda")
            pixel_values = pixel_values.to("cuda")

        generated_ids = model.generate(pixel_values=pixel_values, input_ids=input_ids, max_length=50)
        processed_output = processor.batch_decode(generated_ids, skip_special_tokens=True) 

        processed_output = processed_output[0].replace(question.lower(), '').strip()

        accuracy = accuracy + accuracyVQA(processed_output, answers)

        del input_ids
        del pixel_values
    
    del model
    
    return format_accuracy(accuracy, len(annotations))

"""

Generate the file .csv, containing the accuracy of all the models being tested
The models are hardcoded and since each one slightly differs from the others, has a dedicated function
for inference.

"""

def test_models(source: list, question_type: str):
    annotations_path = None
    imgs_path = None

    csv_name = ('_'.join(source)) + (f'_{question_type}' if question_type is not None else '') + '_test.csv'
    data_mapping = utils.map_data(source)

    annotations_path = data_mapping['annotations']['val']
    imgs_path = data_mapping['images']['val']
    question_path = data_mapping['questions']['val'] if question_type is None else data_mapping['questions'][question_type]['val'] 

    if not os.path.exists(csv_name):
        with open(csv_name, 'w', newline='') as file:
            writer = csv.writer(file)
            dataset = dataset_gen.build_dataset(annotations_path, question_path)

            blip_vqa_accuracy = blip_vqa_test(dataset, annotations_path, imgs_path)
            git_base_vqa_accuracy = git_base_vqa_test(dataset, annotations_path, imgs_path)
            vilt_vqa_accuracy = vilt_vqa_test(dataset, annotations_path, imgs_path)

            field = ["model", "accuracy"]
            writer.writerow(field)
            writer.writerow(["Salesforce/blip-vqa-base", blip_vqa_accuracy])
            writer.writerow(["dandelin/vilt-b32-finetuned-vqa", vilt_vqa_accuracy])
            writer.writerow(["microsoft/git-base-vqav2", git_base_vqa_accuracy]) 
    else:
        print(f'{csv_name} already exists')
            

def main(source: list, question_type: str):
    test_models(source, question_type)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description ='Test the models on available data from VQA2')
    parser.add_argument('--source', 
                    type = utils.list_of_strings, 
                    action='store',
                    help ="real; abstract; abstract,binary")
    
    parser.add_argument('--question_type', 
                    help ="Required in case you are testing pure abstract questions: open_ended or multiple_choice)")
    
    args = parser.parse_args()
    source = args.source
    question_type = args.question_type if hasattr(args, 'question_type') else None
    main(source, question_type)