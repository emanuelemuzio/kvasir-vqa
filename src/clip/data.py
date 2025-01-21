import os
from torch.utils.data import Dataset
import numpy as np
import random
from dotenv import load_dotenv
import torch 
from common.util import init_kvasir_vocab, init_kvasir_vocab_multilabel
from PIL import Image

RANDOM_SEED = int(os.getenv('RANDOM_SEED'))

np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

load_dotenv()  
    
class _Dataset(Dataset): 
    
    def __init__(self, source=[], question=[], answer=[], img_id=[],  base_path='', processor=None):  
        
        self.source = source
        self.question = question
        self.answer = answer
        self.img_id = img_id
        self.base_path = base_path 
        self.processor = processor
        self.config = get_config()
    
    def __len__(self):
        return len(self.source)

    def __getitem__(self, idx):
        
        question = self.question[idx]
        answer = self.answer[idx]
        img_id = self.img_id[idx]
        txt_answer = self.config['id2label'][np.argmax(answer)]
        
        full_path = f"{self.base_path}/{img_id}.jpg" 
        img = Image.open(full_path)

        encoding = self.processor(
            text=txt_answer,
            images=img,
            padding="max_length",
            return_tensors="pt"  
        ) 
        
        encoding['answer'] = torch.tensor(answer)
        encoding['question'] = torch.tensor(question)
        
        return {
            'encoding': encoding,
            'question' : question,
            'answer' : answer
        }
    
def get_config():
    labels = init_kvasir_vocab()
    ids = [i for i in range(len(labels))]
    
    label2id = {label : idx for (label, idx) in zip(labels, ids)}
    id2label = {idx : label for (label, idx) in zip(labels, ids)}
    
    config = {
        "label2id" : label2id,
        "id2label" : id2label
    }
    
    return config

def get_multilabel_config():
    labels = init_kvasir_vocab_multilabel()
    ids = list(labels.values())
    labels = list(labels.keys())
    
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