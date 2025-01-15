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
import shutil
from nltk.translate.bleu_score import sentence_bleu
from nltk.translate import meteor
from nltk import word_tokenize
import pandas as pd
import evaluate
rouge = evaluate.load('rouge')

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
            
            plt.close()
            
 
 
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

def update_best_runs(key='', best_run_path='', runs_path='', del_others=False):
    
    subject = 'feature extractor' if key == 'model' else 'classifier'
    
    logger.info(f"Updating runs for {subject}")
    
    runs_list = []
    
    for run_id in os.listdir(runs_path):
        if os.path.exists(f"{ROOT}/{runs_path}/{run_id}/run.json"):
            data = get_run_info(f"{runs_path}/{run_id}/run.json")
            runs_list.append({
                "model" : data['model'],
                "test_acc" : data['test_acc'],
                "run_id" : run_id
            })
        else:
            logger.info(f"Cleaning {runs_path}/{run_id} directory")
            os.rmdir(f"{ROOT}/{runs_path}/{run_id}")

    best_runs = {}

    for run in runs_list:
        model = run[key]
        test_acc = run['test_acc']
        run_id = run['run_id']

        if model not in best_runs or test_acc > best_runs[model]['test_acc']:
            logger.info(f"New best run with id {run_id} for model {model}")
            best_runs[model] = {'run_id': run_id, 'test_acc': test_acc}
        else:
            if del_others:
                shutil.rmtree(f"{ROOT}/{runs_path}/{run_id}")
 
    with open(f"{ROOT}/{best_run_path}", "w") as f:
        json.dump(best_runs, f, indent=4)
        
def update_best_runs_v2(best_run_path='', runs_path=''):
    
    logger.info(f"Updating best runs in {best_run_path}")
    
    runs_list = []
    
    for run_id in os.listdir(runs_path):
        if os.path.exists(f"{ROOT}/{runs_path}/{run_id}/run.json"):
            data = get_run_info(f"{runs_path}/{run_id}/run.json")
            runs_list.append({
                "model" : data['model'],
                "test_acc" : data['test_acc'],
                "run_id" : run_id
            })
        else:
            logger.info(f"Cleaning {runs_path}/{run_id} directory")
            os.rmdir(f"{ROOT}/{runs_path}/{run_id}")

    best_runs = {}

    for run in runs_list:
        model = run['model']
        test_acc = run['test_acc']
        run_id = run['run_id']

        if model not in best_runs or test_acc > best_runs[model]['test_acc']:
            logger.info(f"New best run with id {run_id} for model {model}")
            best_runs[model] = {'run_id': run_id, 'test_acc': test_acc}
 
    with open(f"{ROOT}/{best_run_path}", "w") as f:
        json.dump(best_runs, f, indent=4)
        
def delete_other_runs(runs_path=''):
    for run_id in os.listdir(runs_path):
        logger.info(f"Deleting run with id: {run_id}")
        shutil.rmtree(f"{ROOT}/{runs_path}/{run_id}")
        

def get_best_feature_extractor_info():
    best_run_path = os.getenv('BEST_RUN_PATH').replace('REPLACE_ME', 'feature_extractor')
    
    f = open(f"{ROOT}/{best_run_path}")
    data = json.load(f)
    
    best_acc = -1
    best_config = {}
    
    for model_name in data.keys():
        if data[model_name]['test_acc'] > best_acc:
            best_acc = data[model_name]['test_acc']
            best_config = {
                'model_name' : model_name,
                'test_acc' : data[model_name]['test_acc'],
                'run_id' : data[model_name]['run_id']
            }
            
    return best_config

def calculate_rouge(candidates, references):
    '''
    candidate, reference: generated and ground-truth sentences
    '''
    scores = rouge.compute(predictions=candidates, references=references)
    return scores

def calculate_bleu(candidate, reference):
    '''
    candidate, reference: generated and ground-truth sentences
    '''
    reference = word_tokenize(reference)
    candidate = word_tokenize(candidate)
    score = sentence_bleu(reference, candidate)
    return score

def calculate_meteor(candidate, reference):
    '''
    candidate, reference: tokenized list of words in the sentence
    '''
    reference = word_tokenize(reference)
    candidate = word_tokenize(candidate)
    meteor_score = round(meteor([candidate],reference), 4)
    return meteor_score

def init_kvasir_vocab():
    f = open(f"{ROOT}/{os.getenv('KVASIR_VQA_VOCABULARY')}")
    return json.load(f)

def init_kvasir_vocab_multilabel():
    f = open(f"{ROOT}/{os.getenv('KVASIR_VQA_VOCABULARY_MULTILABEL')}")
    return json.load(f)

def check_run_type(run_path : str) -> int:
    
    '''
    <> args.run_id is not empty and the folder exists:
        <> the folder contains a checkpoint.pt file:
            - the training continues -> 2
        <> the folder doesnt contain a checkpoint.pt:
            <> the folder contains a model.pt file:
                - only run tests -> 3
            <> the folder doesnt contain neither checkpoint nor model:
                - delete folder, create new one for train and then tests -> 1
    <> the run_id is invalid, the folder doesnt exist:
        - create new folder for train and tests -> 1
    <> args.run_id is empty:
        - generate new id and create new folder -> 1
    '''
    
    if os.path.exists(f"{run_path}"):
        if os.path.exists(f"{run_path}/checkpoint.pt"):
            return 2
        else:
            if os.path.exists(f"{run_path}/model.pt"):
                return 3
            else:
                shutil.rmtree(run_path)
                return 1
    else:
        return 1
                
def generative_report(candidate_list, reference_list):
    
    bleu_score = calculate_bleu(candidate_list, reference_list)
    rouge_scores = calculate_rouge(candidate_list, reference_list)
    meteor_score = calculate_meteor(candidate_list, reference_list)
    rouge1_score = rouge_scores['rouge1']
    rouge2_score = rouge_scores['rouge2']
    rougeL_score = rouge_scores['rougeL']
    rougeLsum_score = rouge_scores['rougeLsum']
    
    metrics = [
        'BLEU',
        'METEOR',
        'ROUGE-1',
        'ROUGE-2',
        'ROUGE-L',
        'ROUGE-L SUM',
    ]
    
    scores = [
        bleu_score,
        meteor_score,
        rouge1_score,
        rouge2_score,
        rougeL_score,
        rougeLsum_score
    ]
    
    return pd.DataFrame({
        "Metric" : metrics,
        "Score" : scores
    })
    
def decorate_prompt(question : str, questions_map : str, strategy : str):
    
    prompt_strategies = {
        "template-1" : "Answer the following question: <q> Options: <o>",
        "cot-1" : " Q: <q> Options: <o>, A: Let’s think step-by-step"
    }
    
    decorated_prompt = prompt_strategies[strategy].replace(
                        '<q>', question
                       ).replace(
                        '<o>', ', '.join(questions_map[question])
                       )
    
    return decorated_prompt



logger = init_logger(logging)