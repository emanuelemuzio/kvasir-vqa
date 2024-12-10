
from datasets import load_dataset
import os
from torchvision.io import read_image
from torch.utils.data import Dataset
import numpy as np
import random
from dotenv import load_dotenv
import torch
from common.util import logger, ROOT, image_transform
from common.prompt_tuning import PromptTuning
import pandas as pd

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
    
    def __init__(self, source=[], question=[], answer=[], img_id=[],  base_path='', prompt=[]):  
        
        self.source = source
        self.question = question
        self.answer = answer
        self.img_id = img_id
        self.base_path = base_path
        self.prompt = prompt
        self.transform = image_transform()
        self.use_prompt = len(self.prompt) > 0
        
        logger.info('Initialized Kvasir VQA Dataset')
    
    def __len__(self):
        return len(self.source)

    def __getitem__(self, idx):
        
        question = self.question[idx]
        answer = self.answer[idx]
        img_id = self.img_id[idx]
        use_prompt = self.use_prompt
        
        if use_prompt:
            question += self.prompt[idx]
        
        base_path = self.base_path
        
        full_path = f"{base_path}/{img_id}.jpg"
        img = read_image(full_path)         

        transformed_image = self.transform(img.float())

        return transformed_image, question, answer
    
    
    
def generate_prompt_dataset() -> None:
    
    try:
        logger.info('Generating questions prompt')  
            
        df = pd.read_csv(os.getenv('KVASIR_VQA_CSV')) 
            
        questions = df['question'].unique().tolist()
        
        prompt_tuning = PromptTuning(os.getenv('PROMPT_TUNING_MODEL'))
            
        prompted_questions = prompt_tuning.generate(question=questions)
            
        logger.info("Finished prompt generation")
        
        df['prompt'] = np.nan
                        
        for question, prompted_question in zip(questions, prompted_questions):
            df.loc[df['question'] == question, 'prompt'] = prompted_question
            
        df.to_csv(os.getenv('KVASIR_VQA_PROMPT_CSV'), index=False)
            
        logger.info("Csv file saved")
    except Exception as e:
        logger.error("An error occurred during prompt generation")
        