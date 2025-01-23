

import sys
sys.path.append('src')

import warnings
warnings.filterwarnings('ignore')

from torchviz import make_dot
import os
import argparse
from dotenv import load_dotenv
import torch
from common.util import ROOT
from custom.model import get_classifier
import pandas as pd
from torch.utils.data import DataLoader
from custom.data import Dataset_
from question_encode.model import get_tokenizer, get_language_model
from feature_extractor.model import init
import torchvision
from torchview import draw_graph
from graphviz import Source

load_dotenv()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(device)

exit(1)
 
if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--run_id', help="Run id")

    args = parser.parse_args()       
    
    run_id = args.run_id  
    
    kvasir_vqa_datapath = f"{ROOT}/{os.getenv('KVASIR_VQA_DATA')}"
    kvasir_vqa_datapath_aug = f"{ROOT}/{os.getenv('KVASIR_VQA_DATA_AUG')}"
    run_path = f"{ROOT}/{os.getenv('CUSTOM_RUNS')}/{run_id}"
    run_json = f"{ROOT}/{os.getenv('CUSTOM_RUNS')}/{run_id}/run.json"
    model = get_classifier(
        'resnet50',
        11, #should be 469
        'concat',
        False
    ).to(device)
    
    df = pd.read_csv(f"{ROOT}/{os.getenv('KVASIR_VQA_CSV_AUG')}")
    df = df[:10]
    df.dropna(axis=0, inplace=True)
    
    y_column = 'answer'
    x_columns = df.columns.to_list()
    x_columns.remove(y_column)
    
    X = df.drop(y_column, axis=1)
    Y = df[y_column]
    
    dataset = Dataset_(
        source=X['source'].to_numpy(), 
        question=X['question'].to_numpy(), 
        answer=Y.to_numpy(), 
        img_id=X['img_id'].to_numpy(), 
        base_path=kvasir_vqa_datapath,
        aug_path=kvasir_vqa_datapath_aug
    )
    
    dataloader = DataLoader(dataset, batch_size=1)
    
    batch = next(iter(dataloader))
    
    img = batch[0]
    question = batch[1]
    answer = batch[2]
    
    weights = torch.load(f"{run_path}/model.pt") 
    
    feature_extractor_run_path = f"{ROOT}/{os.getenv('FEATURE_EXTRACTOR_RUNS')}/{str(23122024203310)}/run.json"
    feature_extractor_weights_path = f"{ROOT}/{os.getenv('FEATURE_EXTRACTOR_RUNS')}/{str(23122024203310)}/model.pt"
    tokenizer = get_tokenizer()
    question_encoder = get_language_model().to(device)
    feature_extractor = init(model_name='resnet50', weights_path=feature_extractor_weights_path).to(device)
    
    inputs = tokenizer(
                        question, 
                        add_special_tokens=True, 
                        return_tensors='pt', 
                        padding='longest', 
                        max_length=15, 
                        truncation=True)
        
    input_ids = inputs['input_ids'].to(device)
    token_type_ids = inputs['token_type_ids'].to(device)
    attention_mask = inputs['attention_mask'].to(device)

    word_embeddings = question_encoder(input_ids, attention_mask = attention_mask, token_type_ids = token_type_ids).last_hidden_state[:,0,:].to(device)
            
    img = img.to(device)
            
    image_feature = feature_extractor(img).squeeze(2).squeeze(-1)

    model.load_state_dict(weights)
    
    # model_graph = draw_graph(model, input_size=(word_embeddings.shape, image_feature.shape), expand_nested=True)
    # model_graph.resize_graph(scale=5.0)
    # model_graph.visual_graph.render(format='png')
    output = model(word_embeddings, image_feature)
        
    dot = make_dot(output, params=dict(list(model.named_parameters())))
    Source(dot).render('test.png')
    dot.format = 'png' 
    dot.render(f'{run_id}.png')