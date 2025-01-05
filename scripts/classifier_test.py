import sys
sys.path.append('src')

import warnings
warnings.filterwarnings('ignore')

import os
import torch
import argparse
from dotenv import load_dotenv
from classifier.model import launch_experiment
from common.util import ROOT, get_run_info, update_best_runs, get_best_feature_extractor_info

load_dotenv() 

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
 
if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--feature_extractor', help="use the run id in order to retrieve automatically the backbone used for the feature extraction")
    parser.add_argument('--architecture', help="'concat', 'hadamard'")
    parser.add_argument('--prompting', help="'1' for prompt aided question, else '0'")
    parser.add_argument('--use_best_fe', help="'1' to use the feature extractor with the highest test accuracy")

    args = parser.parse_args()
    
    run_all = args.run_all == "1"
        
    delete_ckp = args.delete_ckp == "1"
    
    use_best_fe = args.use_best_fe == "1"
    
    if use_best_fe:
        best_fe_info = get_best_feature_extractor_info()
        configs['feature_extractor'] = [best_fe_info['model_name']]
        args.feature_extractor = best_fe_info['run_id']
    
    if delete_ckp:
        if os.path.exists(os.getenv('CLASSIFIER_CHECKPOINT')):
            os.remove(os.getenv('CLASSIFIER_CHECKPOINT'))
            
    if run_all:
        skip_configs = []
        
        for run_id in os.listdir(f"{ROOT}/{os.getenv('CLASSIFIER_RUNS')}"):
            run_json_path = f"{ROOT}/{os.getenv('CLASSIFIER_RUNS')}/{run_id}/run.json"
            if os.path.exists(run_json_path):
                run_info = get_run_info(run_json_path)
                
                skip_configs.append((run_info['feature_extractor'], run_info['architecture'], run_info['prompting']))
                
        for feature_extractor in configs['feature_extractor']:
            for architecture in configs['architecture']:
                for prompting in configs['prompting']:
                                        
                    args.feature_extractor = feature_extractor
                    args.architecture = architecture
                    args.prompting = prompting
                    
                    if not (feature_extractor, architecture, prompting) in skip_configs:                    
                        launch_experiment(args=args, device=device)
    else:
        launch_experiment(args=args, device=device)
    
    turnoff = int(args.turnoff)
    
    del_others = args.del_others == '1'
    
    update_best_runs(
        key='architecture',
        best_run_path=os.getenv('BEST_RUN_PATH').replace('REPLACE_ME','classifier'),
        runs_path=os.getenv('CLASSIFIER_RUNS'),
        del_others=del_others
    )
    
    if turnoff >= 0:
        os.system(f"shutdown /s /t {turnoff}")