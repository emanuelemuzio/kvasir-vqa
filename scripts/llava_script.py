

import sys
sys.path.append('src')

import warnings
warnings.filterwarnings('ignore')

import os
import argparse
from dotenv import load_dotenv
from llava.model import launch_experiment
from common.util import clean_runs, ROOT

load_dotenv()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--model', help="Model name for info purpose")
    parser.add_argument('--turnoff', help="x < 0 (es. -1) for no turnoff, 0 for instant turnoff, x > 0 to plan the turnoff in x seconds") 
    parser.add_argument('--prompting', help="'1' for prompt aided question, else '0'")

    args = parser.parse_args()       
        
    launch_experiment(args=args)
    
    turnoff = int(args.turnoff)
    
    clean_runs(unique_key='generative_report.csv', path=f"{ROOT}/{os.getenv('LLAVA_RUNS')}")
    
    if turnoff >= 0:
        os.system(f"shutdown /s /t {turnoff}")