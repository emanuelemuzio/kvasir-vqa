import sys
sys.path.append('src')

import warnings
warnings.filterwarnings('ignore')

import torch
import argparse
import os
from feature_extractor.model import launch_experiment, ROOT
from common.util import get_run_info, update_best_runs, delete_other_runs, logger

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
    
    logger.info("Starting feature_extractor_finetune script")
    
    parser = argparse.ArgumentParser()
    
    '''
    Brief description of the parameter, followed by the standard/expected value for the standard case.
    '''
    
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
    parser.add_argument('--run_all', help="1 if ALL configurations have to be tested")
    parser.add_argument('--delete_ckp', help="If equals '1', the existing checkpoint will be deleted")
    parser.add_argument('--min_epochs', help='Early stopper min. epochs activation')
    parser.add_argument('--del_others', help='Delete worse runs for the same model')
    parser.add_argument('--tabula_rasa', help='Delete previous runs (CAREFUL WITH THIS)')

    args = parser.parse_args() 
    
    run_all = args.run_all == "1"
    
    delete_ckp = args.delete_ckp == "1"
    
    tabula_rasa = args.tabula_rasa == "1"
    
    if tabula_rasa:
        logger.info("Deleting previous runs")
        delete_other_runs(os.getenv('FEATURE_EXTRACTOR_RUNS'))
    
    if delete_ckp:
        if os.path.exists(os.getenv('FEATURE_EXTRACTOR_CHECKPOINT')):
            logger.info("Deleting existing checkpoint")
            os.remove(os.getenv('FEATURE_EXTRACTOR_CHECKPOINT'))
    
    if run_all:
    
        skip_configs = []
        
        for run_id in os.listdir(f"{ROOT}/{os.getenv('FEATURE_EXTRACTOR_RUNS')}"):
            run_json_path = f"{ROOT}/{os.getenv('FEATURE_EXTRACTOR_RUNS')}/{run_id}/run.json"
            if os.path.exists(run_json_path):
                run_info = get_run_info(run_json_path)
                                
                skip_configs.append((run_info['model'], run_info['freeze'], run_info['aug']))
                
        for model in configs['model']:
            for freeze in configs['freeze']:
                for aug in configs['aug']:
                                        
                    args.model = model
                    args.freeze = freeze
                    args.aug = aug
                    
                    if not (model, freeze, aug) in skip_configs:      
                        logger.info(f"Running the configuration with model: {model}, freeze: {freeze} and aug: {aug}") 
                        launch_experiment(args=args, device=device)
                    else:
                        logger.info(f"Skipping config with model: {model}, freeze: {freeze} and aug: {aug}")
                                
    else:
        logger.info("Launching single experiment")
        launch_experiment(args=args, device=device)  
        
    turnoff = int(args.turnoff)
    
    del_others = args.del_others == '1'
    
    update_best_runs(
        key='model',
        best_run_path=os.getenv('BEST_RUN_PATH').replace('REPLACE_ME','feature_extractor'),
        runs_path=os.getenv('FEATURE_EXTRACTOR_RUNS'),
        del_others=del_others
    )
        
    if turnoff >= 0:
        logger.info(f"Turning off in {turnoff} seconds")
        os.system(f"shutdown /s /t {turnoff}") 