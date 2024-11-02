from datetime import datetime
import logging
import json
import os
import matplotlib.pyplot as plt

ROOT = os.getcwd()

'''
Utility function for logger initialization
'''

def init_logger():
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
    
'''
Generate run ID from timestamp
'''

def generate_run_id():

    logging.info('Generating run id')
    now = datetime.now()
    now = now.strftime("%d%m%Y%H%M%S")
    return now

def format_float(x):
    return "{:.2f}".format(x)

'''
Function used for extracting common subsequences from a list of strings.
In particular, it was mainly used for extracting common subsequences from
kvasir vqa questions, for data analysis purposes.
'''

def get_common_subsequences(strings : list) -> list:

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

'''
Retrieve training run info
'''

def get_run_info(run_path : str):
    f = open(run_path)
    data = json.load(f)

    return data

'''
Plot run info from a model training session using matplotlib functions
'''

def plot_run(base_path, run_id):
    run_path = f"{ROOT}/{base_path}/{run_id}/run.json"
    if os.path.exists(run_path):
        with open(run_path, 'r') as file:
            data = json.load(file)

            actual_epochs = list(range(1, len(data['train_loss']) + 1))

            plt.plot(actual_epochs, data['train_loss'], 'r', label="Train loss")
            plt.plot(actual_epochs, data['val_loss'], 'g', label="Val loss")
            plt.plot(actual_epochs, data['val_acc'], 'b', label="Val acc")
            plt.legend(loc="upper right")

            plt.savefig(f"{base_path}/{run_id}/run.png")

init_logger()