import os
from torchvision.io import read_image
from torch.utils.data import Dataset
import numpy as np
import random
from dotenv import load_dotenv
import torch
from torchvision.transforms import v2
import pandas as pd

RANDOM_SEED = int(os.getenv('RANDOM_SEED'))

np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

load_dotenv()  

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
    
    def __init__(self, source=[], question=[], answer=[], img_id=[],  base_path='', aug_path='', processor=None):  
        
        self.source = source
        self.question = question
        self.answer = answer
        self.img_id = img_id
        self.base_path = base_path
        self.aug_path = aug_path
        self.prompt = []
        self.transform = v2.Compose([
            v2.Resize((384, 640)),  
            v2.ToTensor(),          
        ])
        self.use_prompt = len(self.prompt) > 0 
        self.processor = processor
        
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
        img = self.transform(read_image(full_path))

        encoding = self.processor(
            img, 
            question, 
            padding="max_length", 
            truncation=True, 
            return_tensors="pt"
        )
        
        labels = self.processor.tokenizer.encode(
            answer, max_length= 8, 
            pad_to_max_length=True, 
            return_tensors='pt'
        )
        
        encoding["labels"] = labels
        # remove batch dimension
        for k,v in encoding.items():  
            encoding[k] = v.squeeze()
        return encoding