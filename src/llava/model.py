import sys
sys.path.append('src')

import torch
import os
import pandas as pd
import argparse
import json
from common.util import ROOT, logger, generate_run_id, create_generative_report, decorate_prompt
from llava.data import Dataset_
from dotenv import load_dotenv
from transformers import AutoProcessor, LlavaForConditionalGeneration
from tqdm.auto import tqdm

load_dotenv()      
RANDOM_SEED = int(os.getenv('RANDOM_SEED')) 

def predict(
    model,  
    dataset
    ):
    
    model.eval()
    
    question_list = []
    pred_list = []
    true_list = []
        
    with torch.no_grad():
        for item in tqdm(dataset):
            
            inputs = item['inputs'].to(model.device, torch.float16)
            question = item['question']
            true = item['answer']
            # prompted_question = item['prompted_question']

            generate_ids = model.generate(**inputs, max_new_tokens=100)
            generated_text = dataset.processor.batch_decode(generate_ids, skip_special_tokens=True)[0]
            
            pred = generated_text.split("ASSISTANT:")[-1]
            
            question_list.append(question)
            true_list.append(true)
            pred_list.append(pred)
            
    return question_list, pred_list, true_list
    
def launch_experiment(args : argparse.Namespace) -> None:
    
    questions_map = None
    
    with open(os.getenv('QUESTIONS_MAP')) as f:
        questions_map = json.load(f)
    
    model = LlavaForConditionalGeneration.from_pretrained("llava-hf/llava-1.5-7b-hf", torch_dtype=torch.float16, device_map="auto")
    processor = AutoProcessor.from_pretrained("llava-hf/llava-1.5-7b-hf")
    
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
    
    logger.info("Generating run id")
    run_id = generate_run_id()
      
    run_path = f"{ROOT}/{os.getenv('LLAVA_RUNS')}/{run_id}" 
    os.mkdir(run_path)
    
    logger.info(f'Starting test phase')
    
    question_list, candidate_list, reference_list = predict(
        model=model,  
        dataset=dataset
    ) 
        
    results = create_generative_report(question_list, candidate_list, reference_list)
    results.to_csv(f"{run_path}/generative_report.csv", index=False, sep=";")
        
    logger.info("Saved generative report")
    
    with open(f"{run_path}/run.json", "w") as f:
        config = vars(args)
        config['run_id'] = run_id
        json.dump(config, f)