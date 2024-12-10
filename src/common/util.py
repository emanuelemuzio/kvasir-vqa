from datetime import datetime
import json
import os
import matplotlib.pyplot as plt
import logging
import numpy as np
import random
from dotenv import load_dotenv
import torch
from torchvision.transforms import v2

load_dotenv()

ROOT = os.getcwd() 

RANDOM_SEED = int(os.getenv('RANDOM_SEED'))

np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)



def init_logger(logging):
    
    '''
    Logger inizialization function.
    
    ----------
    Parameters
        logging: package
            Logging package
    ----------

    ------
    Return
        logging: logging
            Configured logger
    ------
    '''
    
    now = datetime.now()
    now = now.strftime("%Y-%m-%d")
    
    logging.basicConfig(
        filename=f"{ROOT}/logs/{now}.log",
        encoding="utf-8",
        filemode="a",
        format="{asctime} - {levelname} - {message}",
        style="{",
        datefmt="%Y-%m-%d %H:%M",
        force=True,
        level=logging.INFO
    )
    
    return logging  
    


def generate_run_id() -> str:
    
    '''
    Generate run ID from timestamp 

    ------
    Return
        run_id: str
            Generated run_id
    ------
    '''
    
    now = datetime.now()
    now = now.strftime("%d%m%Y%H%M%S")
    return now



def format_float(x : float) -> str:
    
    '''
    Float formatting function to .2f format.
    
    ----------
    Parameters
        x: float
            Float to format
    ----------

    ------
    Return
        output: str
            Formatted float
    ------
    '''
    
    return "{:.2f}".format(x)



def get_common_subsequences(strings : list) -> list:
    
    '''
    Function used for extracting common subsequences from a list of strings.
    In particular, it was mainly used for extracting common subsequences from
    kvasir vqa questions, for data analysis purposes.
    
    ----------
    Parameters
        strings: list
            List of strings
    ----------

    ------
    Return
        sorted_subsequences: str
            List of common string subsequences, grouped by freq and ordered alphabetically
    ------
    '''

    subsequences = []

    for string in strings:
        words = string.split(' ')

        word_list = []

        for w in words:
            word_list.append(w)
            sequence = ' '.join(word_list)
            counter = 0

            for s in strings:
                if s.startswith(sequence):
                    counter += 1

            if counter > 1:
                entry = {
                    "subsequence" : sequence,
                    "freq" : counter
                }
                
                if not entry in subsequences:
                    subsequences.append(entry)
        
    frequencies = list(sorted(set(map(lambda x: x['freq'], subsequences)), reverse=True))
    
    sorted_subsequences = []
    
    for f in frequencies:
        grouped = list(sorted(filter(lambda x: x['freq'] == f, subsequences), key=lambda x: len(x['subsequence']), reverse=True))
        
        sorted_subsequences += grouped
        
    return sorted_subsequences 



def get_run_info(run_path : str):
    
    '''
    Retrieve training run info
    
    ----------
    Parameters
        run_path: str
            Path to run.json
    ----------

    ------
    Return
        data: dict
            JSON obj
    ------
    '''
    
    f = open(run_path)
    data = json.load(f)

    return data
 
 

def plot_run(base_path : str, run_id : str) -> None:
    
    '''
    Retrieve training run info
    
    ----------
    Parameters
        base_path: str
            Path to runs folder
        run_id: str
            Run identifier
    ---------- 
    '''
    
    run_path = f"{base_path}/{run_id}/run.json"
    if os.path.exists(run_path):
        with open(run_path, 'r') as file:
            data = json.load(file)

            actual_epochs = list(range(1, len(data['train_loss']) + 1))

            plt.plot(actual_epochs, data['train_loss'], 'r', label="Train loss")
            plt.plot(actual_epochs, data['val_loss'], 'g', label="Val loss")
            plt.plot(actual_epochs, data['val_acc'], 'b', label="Val acc")
            plt.legend(loc="upper right")

            plt.savefig(f"{base_path}/{run_id}/run.png")
            

 
 
 
def id2label(idx : int, classes: list) -> str:
    
    '''
    Function for transforming an id to a label
    
    ----------
    Parameters
        idx: int
            Label id
        classes: list
            list of classes
    ----------

    ------
    Return
        label: str
            label value
    ------
    '''
    
    return classes[idx] 



def id2label_list(idx_list : list, classes : list) -> list:
    
    '''
    Function for transforming a list of ids to a list of labels
    
    ----------
    Parameters
        idx_list: list
            List of ids to transform
        classes: list
            list of classes
    ----------

    ------
    Return
        output: list
            list of ids transformed to labels
    ------
    '''
    
    output = []

    for idx in idx_list:
        output.append(id2label(idx, classes))

    return output



def label2id(label : str, classes: list) -> str:
    
    '''
    Function for transforming a single label to an id
    
    ----------
    Parameters
        label: str
            label to transform
        classes: list
            list of classes
    ----------

    ------
    Return
        id: int
            label transformed to id
    ------
    '''
    
    return classes.index(label)



def label2id_list(label_list : list, classes : list) -> list:
    
    '''
    Function for converting a label list to an id list
    
    ----------
    Parameters
        label_list: str
            label to transform
        classes: list
            list of classes
    ----------

    ------
    Return
        output: list
            labels transformed to ids
    ------
    '''
    
    output = []

    for label in label_list:
        output.append(label2id(label, classes))

    return output
 


def image_transform() -> v2.Compose:
    
    '''
    Preprocessing function, applied before feeding images to the feature extractor.
     
    ------
    Return
        transform: v2.Compose
            List torch transforms that will be applied to the image: 
            - Resize to 224x224
            - Float conversion
            - Normalization
    ------
    '''

    transform = v2.Compose([
        v2.Resize((224,224)),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    return transform 


            
logger = init_logger(logging)