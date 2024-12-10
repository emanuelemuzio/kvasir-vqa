import sys
sys.path.append('src')

import warnings
warnings.filterwarnings('ignore')

import torch
from feature_extractor.model import launch_experiment
import argparse
import os

configs = {
    "model" : [
        "resnet50",
        "resnet101",
        "resnet152",
        "vgg16",
        "vitb16"
    ],
    "freeze" : [
        "1",
        "2"
    ],
    "aug" : [
        "1",
        "0"
    ]
}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--run_all', help="1 if ALL configurations have to be tested")
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
    parser.add_argument('--scheduler', help="'cosine', 'plateau' and 'linear'. write as a csv row for multiple schedulers")
    parser.add_argument('--optimizer', help="'sgd', 'adam' or 'adamw")
    parser.add_argument('--weight_decay', help="")
    parser.add_argument('--model', help="'resnet50', 'resnet101', 'resnet152', 'vgg16' or 'vitb16")
    parser.add_argument('--freeze', help="'0' for inference, '1' for training only the top layer or '2' for training the entire model")
    parser.add_argument('--aug', help="'1' for augmented dataset, else '0'")

    args = parser.parse_args() 
    
    run_all = args.run_all == "1"
    
    if run_all:
        for model in configs['model']:
            for freeze in configs['freeze']:
                for aug in configs['aug']:
                    args.model = model
                    args.freeze = freeze
                    args.aug = aug
                    launch_experiment(args=args, device=device)
    else:
        launch_experiment(args, device)  
        
    turnoff = int(args.turnoff)
        
    if turnoff >= 0:
        os.system(f"shutdown /s /t {turnoff}") 