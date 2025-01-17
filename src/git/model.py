import sys
sys.path.append('src')

import torch
import os
import pandas as pd
import argparse
import json
from common.util import ROOT, logger, generate_run_id, create_generative_report, decorate_prompt
from git.data import Dataset_
from dotenv import load_dotenv
from transformers import AutoProcessor, AutoModelForCausalLM
from tqdm.auto import tqdm

load_dotenv()      
RANDOM_SEED = int(os.getenv('RANDOM_SEED')) 

def predict(
    model, 
    device,
    dataset
    ):
    
    model.eval()
    
    question_list = []
    candidate_list = []
    reference_list = []
        
    with torch.no_grad():
        for item in tqdm(dataset):
            
            input_ids = item['input_ids'].to(device)
            pixel_values = item['pixel_values'].to(device)
            question = item['question']
            reference = item['answer']

            generated_ids = model.generate(pixel_values=pixel_values, input_ids=input_ids, max_length=100)
            
            candidate = dataset.processor.batch_decode(generated_ids[:, input_ids.shape[1]:], skip_special_tokens=True).pop()
                
            question_list.append(question)
            candidate_list.append(candidate)
            reference_list.append(reference)
                
            if device == 'cuda':
                torch.cuda.empty_cache()
                
    return question_list, candidate_list, reference_list
    
def launch_experiment(args : argparse.Namespace, device: str) -> None:
    
    questions_map = None
    
    with open(os.getenv('QUESTIONS_MAP')) as f:
        questions_map = json.load(f)
    
    processor = AutoProcessor.from_pretrained("microsoft/git-base-textvqa")
    model = AutoModelForCausalLM.from_pretrained("microsoft/git-base-textvqa")
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
    
    logger.info('Building dataloaders...')
    
    prompting = None
    
    if args.prompting is not None:
        prompting = args.prompting
        X['question'] = X['question'].apply(lambda x : decorate_prompt(x, questions_map=questions_map, strategy=prompting))
    
    dataset = Dataset_(
        source=X['source'].to_numpy(), 
        question=X['question'].to_numpy(), 
        answer=Y.to_numpy(), 
        img_id=X['img_id'].to_numpy(), 
        base_path=kvasir_vqa_datapath, 
        processor=processor
    )
    
    if device == 'cuda':
        torch.compile(model, 'max-autotune') 
    
    logger.info("Generating run id")
    run_id = generate_run_id()
      
    run_path = f"{ROOT}/{os.getenv('GIT_RUNS')}/{run_id}" 
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