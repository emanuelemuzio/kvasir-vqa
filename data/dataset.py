import json
import torch.utils.data as data
from PIL import Image
import os
import os.path
import numpy as np
import pandas as pd

ANNOTATIONS_PATH = 'data/v2_mscoco_val2014_annotations.json'
COMPLEMENTARY_PAIRS_PATH = 'data/v2_mscoco_val2014_complementary_pairs.json'
QUESTIONS_PATH = 'data/v2_OpenEnded_mscoco_val2014_questions.json'
COCO_PATH = 'data/val2014'

def extract_ids(x):
    return x['image_id']

def load_json(path):
    f = open(path)
    data = json.load(f)
    return data

class VQADataset(data.Dataset):
    def __init__(self, image_ids, questions, question_ids, answers, answer_confidence, answer_ids, answer_types, multiple_choice_answers, question_types):
        self.image_ids = image_ids
        self.questions = questions
        self.question_ids = question_ids
        self.answers = answers
        self.answer_confidence = answer_confidence
        self.answer_ids = answer_ids
        self.answer_types = answer_types
        self.multiple_choice_answers = multiple_choice_answers
        self.question_types = question_types

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        return {
            "image_id" : self.image_ids.iloc[idx],
            "question" : self.questions.iloc[idx],
            "question_id" : self.question_ids.iloc[idx],
            "answer" : self.answers.iloc[idx],
            "answer_confidence" : self.answer_confidence.iloc[idx],
            "answer_id" : self.answer_ids.iloc[idx],
            "answer_type" : self.answer_types.iloc[idx],
            "multiple_choice_answer" : self.multiple_choice_answers.iloc[idx],
            "question_type" : self.question_types.iloc[idx]
        } 

def build_dataframe():

    header = [
        "image_id",
        "question",  
        "question_id", 
        "answer",
        "answer_confidence", 
        "answer_id", 
        "answer_type", 
        "multiple_choice_answer",
        "question_type"
    ]
    
    annotations_file = load_json(ANNOTATIONS_PATH)
    annotations = annotations_file['annotations']

    questions_file = load_json(QUESTIONS_PATH)
    questions = questions_file['questions']

    assert(len(annotations) == len(questions)) 

    data = []
    dataframe = pd.DataFrame(data, columns=header)


    for (annotation, question) in zip (annotations, questions):
        for answer in annotation['answers']:
            data.append([
                question['image_id'],
                question['question'],
                question['question_id'],
                answer['answer'],
                answer['answer_confidence'],
                answer['answer_id'],
                annotation['answer_type'],
                annotation['multiple_choice_answer'],
                annotation['question_type']
            ])
            
    
    dataframe = pd.DataFrame(data, columns=header)

    return dataframe

def build_dataset():
    dataframe = build_dataframe()
    dataset = VQADataset(
        dataframe['image_id'],
        dataframe['question'],
        dataframe['question_id'],
        dataframe['answer'],
        dataframe['answer_confidence'],
        dataframe['answer_id'],
        dataframe['answer_type'],
        dataframe['multiple_choice_answer'],
        dataframe['question_type']
    )

    return dataset