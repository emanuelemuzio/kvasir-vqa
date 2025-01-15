import sys
sys.path.append('src')

import warnings
warnings.filterwarnings('ignore')

import os
import torch
import argparse
from dotenv import load_dotenv
from custom.model import launch_experiment
from common.util import ROOT, get_run_info, update_best_runs, delete_other_runs, get_best_feature_extractor_info

load_dotenv()

configs = {
    'feature_extractor' : [
        os.getenv('RESNET50_BEST_RUN_ID'),
        os.getenv('RESNET101_BEST_RUN_ID'),
        os.getenv('RESNET152_BEST_RUN_ID'),
        os.getenv('VGG16_BEST_RUN_ID'),
        os.getenv('VITB16_BEST_RUN_ID')
    ],
    'architecture' : [
        'hadamard',
        'concat'
    ],
    'prompting' : [
        '0'
    ]
}


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
 
if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--turnoff', help="x < 0 (es. -1) for no turnoff, 0 for instant turnoff, x > 0 to plan the turnoff in x seconds") 
    parser.add_argument('--num_epochs', help="usually 50, 100, 200 epochs")
    parser.add_argument('--batch_size', help="usually 16, 32 or 64")
    parser.add_argument('--lr', help="usually 1e-3")
    parser.add_argument('--momentum', help="mostly 0.9")
    parser.add_argument('--T_max', help="usually 100")
    parser.add_argument('--eta_min', help="usually 100")
    parser.add_argument('--patience', help="usually 0.001")
    parser.add_argument('--min_delta', help="depends on how sensitive you want the early stopper to be")
    parser.add_argument('--mode', help="'min' for Adam optimizer")
    parser.add_argument('--scheduler', help="'cosine', 'plateau' or 'linear'")
    parser.add_argument('--optimizer', help="'sgd', 'adam' or 'adamw")
    parser.add_argument('--weight_decay', help="1e-2 or 1e-3")
    parser.add_argument('--feature_extractor', help="use the run id in order to retrieve automatically the backbone used for the feature extraction")
    parser.add_argument('--architecture', help="'concat', 'hadamard'")
    parser.add_argument('--prompting', help="'1' for prompt aided question, else '0'")
    parser.add_argument('--run_all', help="1 if ALL configurations have to be tested")
    parser.add_argument('--delete_ckp', help="If equals '1', the existing checkpoint will be deleted")
    parser.add_argument('--min_epochs', help='Early stopper min. epochs activation')
    parser.add_argument('--use_best_fe', help="'1' to use the feature extractor with the highest test accuracy")
    parser.add_argument('--use_aug', help="'1' to use augmented data (important for data balance of kvasirvqa dataset)")

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