import sys
sys.path.append('src')

import os
from torchvision.io import read_image
from torch.utils.data import Dataset
import numpy as np
import random
from dotenv import load_dotenv
import torch
from torchvision.transforms import v2

RANDOM_SEED = int(os.getenv('RANDOM_SEED'))

np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

load_dotenv()  

class Dataset_(Dataset):
    
    def __init__(self, 
                source=[], 
                question=[], 
                prompted_question=[],
                answer=[], 
                img_id=[],  
                base_path='', 
                processor=None,
                ):  
        
        self.source = source
        self.question = question
        self.answer = answer
        self.img_id = img_id
        self.base_path = base_path 
        self.processor = processor
        self.prompted_question = prompted_question
        
    def __len__(self):
        return len(self.source)

    def __getitem__(self, idx):
        
        question = self.question[idx]
        answer = self.answer[idx]
        img_id = self.img_id[idx]
        prompted_question = self.prompted_question[idx]
        
        full_path = f"{self.base_path}/{img_id}.jpg" 
        img = read_image(full_path)
        
        conversation = [
            {
                "role" : "user",
                "content" : [
                    {"type" : "image"},
                    {"type" : "text", "text" : prompted_question} 
                ]
            }
        ]
        
        prompt = self.processor.apply_chat_template(conversation, add_generation_prompt=True)
        inputs = self.processor(images=[img], text=[prompt], padding=True, return_tensors="pt")
        
        return {
            "inputs" : inputs,
            "answer" : answer,
            "question" : question
            # "prompted_question" : prompted_question
        }