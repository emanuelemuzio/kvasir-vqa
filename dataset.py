import torch.utils.data as data
import pandas as pd
from utils import load_json
from tqdm.auto import tqdm

""""

The dataset used for the thesis was VQA 2.0, which consists of both images and JSON
files containing information about the question, the image and the answers provided by users

"""

"""

Torch Dataset used for loading informations about the pairs of images and questions 

"""

class VQADataset(data.Dataset):
    def __init__(
            self, 
            image_ids, 
            questions, 
            question_ids, 
            # answers, 
            # answer_confidence, 
            # answer_ids, 
            # answer_types,
            # multiple_choice_answers, 
            # question_types
            ):
        self.image_ids = image_ids
        self.questions = questions
        self.question_ids = question_ids
        # self.answers = answers
        # self.answer_confidence = answer_confidence
        # self.answer_ids = answer_ids
        # self.answer_types = answer_types
        # self.multiple_choice_answers = multiple_choice_answers
        # self.question_types = question_types

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        return {
            "image_id" : self.image_ids.iloc[idx],
            "question" : self.questions.iloc[idx],
            "question_id" : self.question_ids.iloc[idx]
            # "answer" : self.answers.iloc[idx],
            # "answer_confidence" : self.answer_confidence.iloc[idx],
            # "answer_id" : self.answer_ids.iloc[idx],
            # "answer_type" : self.answer_types.iloc[idx],
            # "multiple_choice_answer" : self.multiple_choice_answers.iloc[idx],
            # "question_type" : self.question_types.iloc[idx]
        } 
    
"""

Function which purpose is to retrieve data from the JSON files.
If include_answers is True, the resulting DataFrame will contain a row for each 
answer provided by users.
If it is False, every row will only contain information about the question, question id and image id.
Either way, it returns a pandas Dataframe.

"""


def build_dataframe(include_answers: bool, annotations_path: str, question_path) -> pd.DataFrame:

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
    ] if include_answers else [
        "image_id",
        "question",  
        "question_id"
    ]
    
    annotations_file = load_json(annotations_path)
    annotations = annotations_file['annotations']

    questions_file = load_json(question_path)
    questions = questions_file['questions']

    assert(len(annotations) == len(questions)) 

    data = []

    # Considero solo risposte con confidence maybe o si?
    if not include_answers:
        for question in tqdm(questions, desc='Looping questions'):
            data.append([
                question['image_id'],
                question['question'],
                question['question_id'],
            ])
    else:
        for (annotation, question) in tqdm(zip(annotations, questions), desc='Looping questions and annotations'):
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

def build_dataset(annotations_path: str, questions_path: str):
    dataframe = build_dataframe(False, annotations_path, questions_path)
    dataset = VQADataset(
        dataframe['image_id'],
        dataframe['question'],
        dataframe['question_id'],
        # df_grouped_by_question['answer'],
        # df_grouped_by_question['answer_confidence'],
        # df_grouped_by_question['answer_id'],
        # df_grouped_by_question['answer_type'],
        # df_grouped_by_question['multiple_choice_answer'],
        # df_grouped_by_question['question_type']
    )

    return dataset