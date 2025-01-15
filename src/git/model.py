import sys
sys.path.append('src')

import torch
import os
import pandas as pd
import argparse
import json
from common.util import ROOT, logger, generate_run_id, generative_report, decorate_prompt
from git.data import Dataset_
from dotenv import load_dotenv
from transformers import AutoProcessor, AutoModelForCausalLM

load_dotenv()      
RANDOM_SEED = int(os.getenv('RANDOM_SEED')) 

def predict(
    model, 
    device,
    dataset
    ):
    
    model.eval()
    
    candidate_list = []
    reference_list = []
        
    with torch.no_grad():
        for item in dataset:
            
            input_ids = item['input_ids'].to(device)
            pixel_values = item['pixel_values'].to(device)
            attention_masked = item['attention_mask'].to(device)
            labels = item['labels'].to(device)

            outputs = model.generate(
                input_ids=input_ids,
                pixel_values=pixel_values,
                attention_mask=attention_masked,
                labels=labels
            )
            
            generated_text = dataset.processor.decode(outputs[0], skip_special_tokens=True)
                
            del(batch)
            if device == 'cuda':
                torch.cuda.empty_cache()
                
    return candidate_list, reference_list
    
def launch_experiment(args : argparse.Namespace, device: str) -> None:
    
    questions_map = None
    
    with open(os.getenv('QUESTIONS_MAP')) as f:
        questions_map = json.load(f)
    
    processor = AutoProcessor.from_pretrained("microsoft/git-base-msrvtt-qa")
    model = AutoModelForCausalLM.from_pretrained("microsoft/git-base-msrvtt-qa")
    model.to(device)
    
    logger.info(f"Launching experiment with configuration: {args}")
    
    logger.info("Generating run id")
  
    kvasir_vqa_datapath = f"{ROOT}/{os.getenv('KVASIR_VQA_DATA')}"
    
    df = pd.read_csv(f"{ROOT}/{os.getenv('KVASIR_VQA_CSV')}") 
    
    logger.info("Dataset retrieved")
        
    df.dropna(axis=0, inplace=True)
    
    y_column = 'answer'
    x_columns = df.columns.to_list()
    x_columns.remove(y_column)
    
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
    
    candidate_list, reference_list = predict(
        model=model, 
        device=device,
        dataset=dataset
    ) 
        
    results = generative_report(candidate_list, reference_list)
    
    results.to_csv(f"{run_path}/generative_report.csv", index=False)
        
    logger.info("Saved generative report")
    
    with open(f"{run_path}/run.json", "w") as f:
        config = vars(args)
        config['run_id'] = run_id
        json.dump(config, f)