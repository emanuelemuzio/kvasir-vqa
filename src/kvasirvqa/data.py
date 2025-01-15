
from datasets import load_dataset
import os
from torchvision.io import read_image
from torch.utils.data import Dataset
import numpy as np
import random
from dotenv import load_dotenv
import torch
from common.util import logger, ROOT, image_transform
from common.prompting import PromptTuning
import pandas as pd
from PIL import Image
from random import randint as rand
from torchvision.transforms import v2

RANDOM_SEED = int(os.getenv('RANDOM_SEED'))

np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

load_dotenv()  

 

def load_kvasir_vqa() -> any:
    
    '''
    Download the KVASIR-VQA dataset, splitted in the metadata.csv file and the imgs
    
    ------
    Return
        dataset:
            HuggingFace dataset
    ------
    '''
    
    logger.info("Downloading KvasirVQA metadata and imgs")
    
    return load_dataset(f"{os.getenv('KVASIR_VQA_DATASET')}")

  

def retrieve_kvasir_vqa_dataset() -> None:
    
    '''
    Utility function for retrieving the original Kvasir VQA dataset using the HuggingFace Loader.
    Hyper Kvasir and Kvasir Instrument classes are then remapped to match the classes present in Kvasir VQA.
    '''
    
    dataset = load_kvasir_vqa()
    dataframe = dataset['raw'].select_columns(['source', 'question', 'answer', 'img_id']).to_pandas()
    
    if not os.path.exists(f"{ROOT}/{os.getenv('KVASIR_VQA_METADATA')}/metadata.csv"):
        logger.info('Retrieving metadata.csv for Kvasir VQA')
        dataframe.to_csv(f"{ROOT}/{os.getenv('KVASIR_VQA_METADATA')}/metadata.csv", index=False)
    else:
        logger.info('Metadata file already exists')
        
    if not os.path.exists(f"{ROOT}/{os.getenv('KVASIR_VQA_DATA')}"):
        logger.info('Retrieving Kvasir VQA data')
        os.makedirs(f"{ROOT}/{os.getenv('KVASIR_VQA_DATA')}", exist_ok=True)
        for i, row in dataframe.groupby('img_id').nth(0).iterrows(): # for images
            dataset['raw'][i]['image'].save(f"{ROOT}/{os.getenv('KVASIR_VQA_DATA')}/{row['img_id']}.jpg")
    else:
        logger.info('Kvasir VQA data already exists')



class Dataset_(Dataset):
    
    '''
    Dataset class implementation for the Kvasir VQA Classifier.
        
    ----------
    Parameters
        source: list
            list of source paths to images
        question: list
            questions paired to images
        answer: list
            answer to questions
        img_id: list
            image identifier
        base_path: str
            base path to the correct folder
        prompt: str
            prompt that decorates the question
    ----------

    ------
    Return
        transformed_image: tensor
            Preprocessed image
        question: str
            question posed about the image, prompt tuned or not
        answer: str
            self explanatory at this point
    ------
    '''
    
    def __init__(self, source=[], question=[], answer=[], img_id=[],  base_path='', aug_path=''):  
        
        self.source = source
        self.question = question
        self.answer = answer
        self.img_id = img_id
        self.base_path = base_path
        self.aug_path = aug_path
        self.prompt = []
        self.transform = image_transform()
        self.use_prompt = len(self.prompt) > 0 
        
    def add_prompts(self, prompt=[]):
        self.prompt = prompt
    
    def __len__(self):
        return len(self.source)

    def __getitem__(self, idx):
        
        question = self.question[idx]
        answer = self.answer[idx]
        img_id = self.img_id[idx]
        use_prompt = self.use_prompt
        
        if use_prompt:
            question += self.prompt[idx]
        
        full_path = f"{self.aug_path}/{img_id}.jpg" if 'aug' in img_id else f"{self.base_path}/{img_id}.jpg" 
        img = read_image(full_path)         

        transformed_image = self.transform(img.float())

        return transformed_image, question, answer
    
    
    
def generate_prompt_dataset() -> None:
    
    try:
        logger.info('Generating questions prompt')  
            
        df = pd.read_csv(os.getenv('KVASIR_VQA_CSV')) 
        df_aug = pd.read_csv(os.getenv('KVASIR_VQA_CSV_AUG')) 
            
        questions = df['question'].unique().tolist()
        questions_aug = df['question'].unique().tolist()
        
        prompt_tuning = PromptTuning(os.getenv('PROMPT_TUNING_MODEL'))
            
        prompted_questions = prompt_tuning.generate(question=questions)
        prompted_questions_aug = prompt_tuning.generate(question=questions_aug)
            
        logger.info("Finished prompt generation")
        
        df['prompt'] = np.nan
        df_aug['prompt'] = np.nan
                        
        for question, prompted_question in zip(questions, prompted_questions):
            df.loc[df['question'] == question, 'prompt'] = prompted_question
        
        for question_aug, prompted_question_aug in zip(questions_aug, prompted_questions_aug):
            df_aug.loc[df_aug['question'] == question_aug, 'prompt'] = prompted_question_aug
            
        df.to_csv(os.getenv('KVASIR_VQA_PROMPT_CSV'), index=False)
        df_aug.to_csv(os.getenv('KVASIR_VQA_PROMPT_CSV_AUG'), index=False)
            
        logger.info("Csv file saved")
    except Exception as e:
        logger.error("An error occurred during prompt generation")
        
def augment_image(src : str, code : str, num : int) -> list:
    
    '''
    ------
    Parameters
        src: str
            Full path to image
        code: str
            Image unique identifier
        num: int
            Number of augmentations to perform per image
    ------
    
    ------
    Return
        data: list
            List that contains paths to all new images generated
    ------
    '''

    data = []

    random_rotations = []
    
    img = Image.open(src)

    for i in range(num):
        r_min = None
        r_max = None

        while True:
            r_min = rand(0, 180)
            r_max = rand(180, 270)

            if (r_min, r_max) not in random_rotations:
                random_rotations.append((r_min, r_max))
                break
        
        transform = v2.Compose([
                    v2.RandomHorizontalFlip(),
                    v2.RandomRotation((r_min, r_max))])
        
        aug = transform(img)

        new_code = f"{code}-aug-{i + 1}"

        new = f"{ROOT}/{os.getenv('KVASIR_VQA_DATA_AUG')}/{new_code}.jpg"

        aug.save(new)
        data.append(new_code)

    return data

def augment_image_where(src : str, code : str, num : int) -> list:
    
    '''
    Function that augment images from KvasirVQA that do not require rotation because
    the answer related to that is a positional one.
    ------
    Parameters
        src: str
            Full path to image
        code: str
            Image unique identifier
        num: int
            Number of augmentations to perform per image
    ------
    
    ------
    Return
        data: list
            List that contains paths to all new images generated
    ------
    '''

    data = []

    transform = v2.Compose([
        v2.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05),
        v2.GaussianBlur(kernel_size=(3, 3), sigma=(0.1, 0.5)),  
    ])
    
    img = Image.open(src)

    for i in range(num): 
        
        aug = transform(img)

        new_code = f"{code}-aug-{i + 1}"

        new = f"{ROOT}/{os.getenv('KVASIR_VQA_DATA_AUG')}/{new_code}.jpg"

        aug.save(new)
        data.append(new_code)

    return data

def augment_kvasir_vqa(df : pd.DataFrame, answ_list : list, threshold : int):
    
    header = ['source', 'question', 'answer', 'img_id']
    rows = []
    
    base_path = f"{ROOT}/{os.getenv('KVASIR_VQA_DATA')}"
    
    for a in answ_list:
        
        data = []
        
        filter_row = df[(df['answer'] == a)]
        src = f"{base_path}/{filter_row['img_id'].iloc[0]}.jpg"
        code = filter_row['img_id'].iloc[0]
        question = filter_row['question'].iloc[0]
        source_cat = filter_row['source'].iloc[0]
        
        if 'Where' in question:
            data.extend(augment_image_where(src=src, code=code, num=threshold-len(filter_row)))
        else:
            data.extend(augment_image(src=src, code=code, num=threshold-len(filter_row)))
            
        for d in data:
            rows.append([source_cat, question, a, d])
            
    df_aug = pd.DataFrame(data=rows, columns=header)

    return df_aug