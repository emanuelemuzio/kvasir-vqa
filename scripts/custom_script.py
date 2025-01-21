import sys
sys.path.append('src')

import warnings
warnings.filterwarnings('ignore')

import os
import torch
import argparse
from dotenv import load_dotenv
from custom.model import launch_experiment
from custom.multilabel import launch_experiment as ml_launch_experiment
from common.util import get_best_feature_extractor_info

load_dotenv() 

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = 'cpu'

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
    parser.add_argument('--delete_ckp', help="If equals '1', the existing checkpoint will be deleted")
    parser.add_argument('--min_epochs', help='Early stopper min. epochs activation')
    parser.add_argument('--use_best_fe', help="'1' to use the feature extractor with the highest test accuracy")
    parser.add_argument('--use_aug', help="'1' to use augmented data (important for data balance of kvasirvqa dataset)")
    parser.add_argument('--format', help="'multilabel'")
    parser.add_argument('--step_size', help="")
    parser.add_argument('--gamma', help="")
    parser.add_argument('--run_id', help="")
    parser.add_argument('--del_others', help="")

    args = parser.parse_args()
    
    data_format = args.format or None
    
    use_best_fe = args.use_best_fe == "1"
    
    if use_best_fe:
        best_fe_info = get_best_feature_extractor_info()
        args.feature_extractor = best_fe_info['run_id']
        
    if data_format == 'multilabel':
        ml_launch_experiment(args=args, device=device)
    else:
        launch_experiment(args=args, device=device)
    
    turnoff = int(args.turnoff)
    
    # update_best_runs(
    #     key='architecture',
    #     best_run_path=os.getenv('BEST_RUN_PATH').replace('REPLACE_ME','custom'),
    #     runs_path=os.getenv('CUSTOM_RUNS'),
    #     del_others=del_others
    # )
    
    if turnoff >= 0:
        os.system(f"shutdown /s /t {turnoff}")