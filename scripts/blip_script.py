

import sys
sys.path.append('src')

import warnings
warnings.filterwarnings('ignore')

import os
import argparse
from dotenv import load_dotenv
from blip.model import launch_experiment

load_dotenv()

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = 'cpu'
 
if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--model', help="Model name for info purpose")
    parser.add_argument('--turnoff', help="x < 0 (es. -1) for no turnoff, 0 for instant turnoff, x > 0 to plan the turnoff in x seconds") 
    parser.add_argument('--prompting', help="'1' for prompt aided question, else '0'")

    args = parser.parse_args()       
        
    launch_experiment(args=args, device=device)
    
    turnoff = int(args.turnoff)
    
    if turnoff >= 0:
        os.system(f"shutdown /s /t {turnoff}")