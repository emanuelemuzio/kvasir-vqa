import os
from torchvision.io import read_image
from torch.utils.data import Dataset
import numpy as np
import random
from dotenv import load_dotenv
import torch
from common.util import ROOT, image_transform
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
    
    def __init__(self, source=[], question=[], answer=[], img_id=[],  base_path='', aug_path='', processor=None, config=None):  
        
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
        self.config = config
        
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
            images=img,
            text=question,
            padding="max_length",   
            truncation=True,    
            return_tensors="pt"  
        ) 
        
        for k,v in encoding.items():
          encoding[k] = v.squeeze()
        
        encoding['answer'] = answer
        
        targets = torch.zeros(len(self.config.id2label))
        targets[self.config.label2id[answer]] = 1
        encoding["labels"] = targets
        encoding["target"] = self.config.label2id[answer]

        return encoding
    
def get_config():
    dataset = pd.read_csv(os.getenv('KVASIR_VQA_CSV'))
    dataset.dropna(inplace=True)
    
    labels = dataset['answer'].unique()
    ids = [i for i in range(len(labels))]
    
    label2id = {label : idx for (label, idx) in zip(labels, ids)}
    id2label = {idx : label for (label, idx) in zip(labels, ids)}
    
    config = {
        "architectures": ["ViltForVisualQuestionAnswering"],
        "attention_probs_dropout_prob": 0.0,
        "hidden_act": "gelu",
        "hidden_dropout_prob": 0.0,
        "hidden_size": 768,
        "id2label": id2label,
        "image_size": 384,
        "initializer_range": 0.02,
        "intermediate_size": 3072,
        "label2id": label2id,
        "layer_norm_eps": 1e-12,
        "max_image_length": -1,
        "max_position_embeddings": 40,
        "modality_type_vocab_size": 2,
        "model_type": "vilt",
        "num_attention_heads": 12,
        "num_channels": 3,
        "num_hidden_layers": 12,
        "num_images": -1,
        "patch_size": 32,
        "qkv_bias": True,
        "tie_word_embeddings": False,
        "torch_dtype": "float32",
        "transformers_version": "4.45.0",
        "type_vocab_size": 2,
        "vocab_size": 30522
    }
    
    return config