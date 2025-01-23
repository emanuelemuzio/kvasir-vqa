import sys
sys.path.append('src')

import torch
import os
import pandas as pd
import argparse
import json
from common.util import ROOT, logger, generate_run_id, create_generative_report, decorate_prompt
from blip.data import Dataset_
from dotenv import load_dotenv
from transformers import BlipProcessor, BlipForQuestionAnswering, logging
from tqdm.auto import tqdm

logging.set_verbosity_error()
logging.set_verbosity(50)  

load_dotenv()      
RANDOM_SEED = int(os.getenv('RANDOM_SEED'))
 
def predict(
    model, 
    device,
    dataset
    ):
    
    question_list = []
    candidate_list = []
    reference_list = []
        
    for item in tqdm(dataset):
            
        encoding = item['encoding'].to(device)
        answer = item['answer']
        question = item['question']
            
        out = model.generate(**encoding)
        generated_text = dataset.processor.decode(out[0], skip_special_tokens=True)
            
        question_list.append(question)
        reference_list.append(answer)
        candidate_list.append(generated_text)
                
        if device == 'cuda':
            torch.cuda.empty_cache()
                
    return question_list, candidate_list, reference_list
    
def launch_experiment(args : argparse.Namespace, device: str) -> None:
    
    questions_map = None
    
    with open(os.getenv('QUESTIONS_MAP')) as f:
        questions_map = json.load(f)
        f.close()
    
    model = BlipForQuestionAnswering.from_pretrained("Salesforce/blip-vqa-base")
    processor = BlipProcessor.from_pretrained("Salesforce/blip-vqa-base")
    model.to(device)
    
    logger.info(f"Launching experiment with configuration: {args}")
    
    logger.info("Generating run id")
  
    kvasir_vqa_datapath = f"{ROOT}/{os.getenv('KVASIR_VQA_DATA')}"
    
    df = pd.read_csv(f"{ROOT}/{os.getenv('KVASIR_VQA_CSV_CLEAN')}") 
    
    logger.info("Dataset retrieved")
        
    df.dropna(axis=0, inplace=True)
    
    y_column = 'answer'
    
    X = df.drop(y_column, axis=1)
    Y = df[y_column]
    
    prompting = None
    
    if args.prompting is not None:
        prompting = args.prompting
        X['prompted_question'] = X['question'].apply(lambda x : decorate_prompt(x, questions_map=questions_map, strategy=prompting))
    
    dataset = Dataset_(
        source=X['source'].to_numpy(), 
        question=X['question'].to_numpy(), 
        prompted_question=X['prompted_question'].to_numpy(),
        answer=Y.to_numpy(), 
        img_id=X['img_id'].to_numpy(), 
        base_path=kvasir_vqa_datapath, 
        processor=processor
    ) 
    
    if device == 'cuda':
        torch.compile(model, 'max-autotune')
    
    logger.info("Generating run id")
    run_id = generate_run_id()
      
    run_path = f"{ROOT}/{os.getenv('BLIP_RUNS')}/{run_id}" 
    os.mkdir(run_path)
    
    logger.info(f'Starting test phase')
    
    question_list, candidate_list, reference_list = predict(
        model=model, 
        device=device,
        dataset=dataset
    ) 
        
    results = create_generative_report(question_list, candidate_list, reference_list)
    results.to_csv(f"{run_path}/generative_report.csv", index=False, sep=";")
        
    logger.info("Saved generative report")
    
    with open(f"{run_path}/run.json", "w") as f:
        config = vars(args)
        config['run_id'] = run_id
        json.dump(config, f)