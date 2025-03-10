

import sys
sys.path.append('src')

import warnings
warnings.filterwarnings('ignore')

import os
import torch
import argparse
from dotenv import load_dotenv
from vilt.model import launch_experiment
from vilt.multilabel import launch_experiment as ml_launch_experiment
from common.util import logger

load_dotenv()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
 
if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--model', help="Model name for info purpose")
    parser.add_argument('--turnoff', help="x < 0 (es. -1) for no turnoff, 0 for instant turnoff, x > 0 to plan the turnoff in x seconds") 
    parser.add_argument('--num_epochs', help="")
    parser.add_argument('--batch_size', help="")
    parser.add_argument('--lr', help="")
    parser.add_argument('--momentum', help="0.9")
    parser.add_argument('--T_max', help="100")
    parser.add_argument('--eta_min', help="100")
    parser.add_argument('--patience', help="10")
    parser.add_argument('--min_delta', help="depends on how sensitive you want the early stopper to be")
    parser.add_argument('--mode', help="'min' for Adam optimizer")
    parser.add_argument('--scheduler', help="'cosine', 'plateau' or 'linear'")
    parser.add_argument('--optimizer', help="'sgd', 'adam' or 'adamw")
    parser.add_argument('--weight_decay', help="")
    parser.add_argument('--step_size', help="StepLR parameter")
    parser.add_argument('--gamma', help="LR reduction parameter for StepLR and LinearLR")
    parser.add_argument('--prompting', help="'1' for prompt aided question, else '0'")
    parser.add_argument('--delete_ckp', help="If equals '1', the existing checkpoint will be deleted")
    parser.add_argument('--min_epochs', help='')
    parser.add_argument('--use_aug', help="'1' to use augmented data (important for data balance of kvasirvqa dataset)")
    parser.add_argument('--run_id', help="Continue the training if checkpoint.pt is present in the run folder, else just run the tests for it")
    parser.add_argument('--model_mode', help="leave empty for normal mode, 'multilabel' if you want to use multilabel transformed metadata file")

    args = parser.parse_args()       
    
    mode = args.model_mode or None
        
    if mode != 'multilabel':
        launch_experiment(args=args, device=device)        
    else:
        ml_launch_experiment(args=args, device=device)
    
        logger.info(f"Updating best runs")
    
    turnoff = int(args.turnoff)
    
    if turnoff >= 0:
        os.system(f"shutdown /s /t {turnoff}")