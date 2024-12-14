import sys
sys.path.append('src')

import warnings
warnings.filterwarnings('ignore')

import os
import torch
import argparse
from dotenv import load_dotenv
from common.util import get_run_info, generate_run_id, ROOT
from classifier.model import launch_experiment

load_dotenv()

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

    args = parser.parse_args()

    launch_experiment(args=args, device=device)
    
    turnoff = int(args.turnoff)
    
    if turnoff >= 0:
        os.system(f"shutdown /s /t {turnoff}")