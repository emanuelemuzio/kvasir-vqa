import sys
sys.path.append('src')

import warnings
warnings.filterwarnings('ignore')

import torch
from feature_extractor.model import launch_experiment
import argparse
import os

'''
The configs dict is going to be useful only if the 'run_all' argument is 1, either way
it is pointless. It contains all the possible macro combinations of parameters.
The rest of the hyperparams will be specified from the other arguments.
'''

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
    
    '''
    Brief description of the parameter, followed by the standard/expected value for the standard case.
    '''
    
    parser.add_argument('--run_all', help="1 if ALL configurations have to be tested")
    parser.add_argument('--turnoff', help="x < 0 for no turn off, else turn off in x seconds") 
    parser.add_argument('--num_epochs', help="Training epochs, 200")
    parser.add_argument('--batch_size', help="Batch size, 32")
    parser.add_argument('--lr', help="Learning rate, 0.0005")
    parser.add_argument('--momentum', help="Momentum for SGD, 0.9")
    parser.add_argument('--T_max', help="CosineAnnealingLR, 100")
    parser.add_argument('--eta_min', help="CosineAnnealingLR, 0.001")
    parser.add_argument('--patience', help="Early Stopper patience, 8")
    parser.add_argument('--min_delta', help="Early Stopper sensitivity, 0.005")
    parser.add_argument('--mode', help="Adam optimizer mode, 'min'")
    parser.add_argument('--scheduler', help="'cosine', 'plateau' and 'linear'. write as a csv row for multiple schedulers, 'cosine,plateau,linear'")
    parser.add_argument('--optimizer', help="'sgd', 'adam' or 'adamw")
    parser.add_argument('--weight_decay', help="Weight decay parameter, 0.0001")
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