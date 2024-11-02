import os
import torch
import pandas as pd
import json
from dotenv import load_dotenv
from torch import optim
from torch.nn import CrossEntropyLoss
import torch.optim as optim 
import shutil
import logging
from ..src.model import EarlyStopper
from ..src.util import get_run_info, generate_run_id, ROOT, init_logger, plot_run
from ..src.dataset import KvasirVQA, df_train_test_split
from ..src.model import get_vqa_classifier, classifier_evaluate, get_tokenizer, get_language_model, init_feature_extractor
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR, LinearLR
import argparse
from sklearn.preprocessing import LabelEncoder

init_logger()
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

    args = parser.parse_args()
    
    feature_extractor_run_path = f"{ROOT}/{os.getenv('FEATURE_EXTRACTOR_RUNS')}/{args.feature_extractor}/run.json"
    feature_extractor_weights_path = f"{ROOT}/{os.getenv('FEATURE_EXTRACTOR_RUNS')}/{args.feature_extractor}/model.pt"
    
    feature_extractor_name = get_run_info(run_path=feature_extractor_run_path)['model'] 
    
    run_id = generate_run_id()

    turnoff = int(args.turnoff)

    logging.info('Recovering classifier model...')

    df = pd.read_csv(f"{ROOT}/{os.getenv('KVASIR_VQA_CSV')}")  
    
    df.dropna(axis=0, inplace=True)
    
    train_set, val_set = df_train_test_split(df, 0.3) 
    val_set, test_set = df_train_test_split(df, 0.5) 

    logging.info('Building dataloaders...')
    
    kvasir_vqa_datapath = f"{ROOT}/{os.getenv('KVASIR_VQA_DATA')}"
    
    train_dataset = KvasirVQA(
        train_set['source'].to_numpy(), 
        train_set['question'].to_numpy(), 
        train_set['answer'].to_numpy(), 
        train_set['img_id'].to_numpy(),
        
        )
    
    val_dataset = KvasirVQA(
        val_set['source'].to_numpy(), 
        val_set['question'].to_numpy(), 
        val_set['answer'].to_numpy(), 
        val_set['img_id'].to_numpy(),
        kvasir_vqa_datapath
        )
    
    test_dataset = KvasirVQA(
        test_set['source'].to_numpy(), 
        test_set['question'].to_numpy(), 
        test_set['answer'].to_numpy(), 
        test_set['img_id'].to_numpy(),
        kvasir_vqa_datapath
        ) 
    
    answers = list(set(df['answer']))
    
    answer_encoder = LabelEncoder().fit(answers)
    
    model = get_vqa_classifier(feature_extractor_name=feature_extractor_name, vocabulary_size=len(answers))
    
    model = model.to(device)
    
    tokenizer = get_tokenizer()
    question_encoder = get_language_model().to(device)
    feature_extractor = init_feature_extractor(model_name=feature_extractor_name, weights_path=feature_extractor_weights_path).to(device)

    if device == 'cuda':
        torch.compile(model, 'max-autotune')

    logging.info('Declaring hyper parameters')
    
    # Train run hyper parameters

    criterion = CrossEntropyLoss()

    # Training loop

    num_epochs = int(args.num_epochs) or 100
    batch_size = int(args.batch_size) or 32
    lr = float(args.lr) or 0.01
    momentum = float(args.momentum) or 0.9
    weight_decay = float(args.weight_decay) or 1e-4
    
    # Cosine Annealing LR params

    T_max = int(args.T_max) or 100
    eta_min = float(args.eta_min) or 0.001
    
    # ReduceLROnPlateau params

    mode = args.mode or 'min'

    patience = int(args.patience) or 5
    min_delta = float(args.min_delta) or 0.01
    
    # Define optimizer, scheduler and early stop

    optimizer = None

    if args.optimizer == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum)
    elif args.optimizer == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif args.optimizer == 'adamw':
        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        
    early_stopper = EarlyStopper(patience=patience, min_delta=min_delta)

    scheduler = []
    
    scheduler_names = args.scheduler.split(',')

    for name in scheduler_names:
        if args.scheduler == 'plateau':
            scheduler.append(ReduceLROnPlateau(optimizer=optimizer, mode=mode))
        elif args.scheduler == 'cosine':
            scheduler.append(CosineAnnealingLR(optimizer=optimizer, T_max=T_max, eta_min=eta_min))
        elif args.scheduler == 'linear':
            scheduler.append(LinearLR(optimizer=optimizer))
        
    max_length = int(os.getenv('MAX_QUESTION_LENGTH'))

    train_loss_ckp = None
    train_acc_ckp = None
    val_loss_ckp = None
    val_acc_ckp = None
    best_acc_ckp = None 
    start_epoch = 1

    run_path = None 
    
    if os.path.exists(f"{ROOT}/{os.getenv('CLASSIFIER_CHECKPOINT')}"):
        checkpoint = torch.load(f"{ROOT}/{os.getenv('CLASSIFIER_CHECKPOINT')}", weights_only=True)
        model.load_state_dict(checkpoint['model_state_dict'])
        start_epoch = checkpoint['num_epochs']
        train_loss_ckp = checkpoint['train_loss']
        train_acc_ckp = checkpoint['train_acc']
        val_loss_ckp = checkpoint['val_loss']
        val_acc_ckp = checkpoint['val_acc']
        best_acc_ckp = checkpoint['best_acc']
        run_id = checkpoint['run_id']
        run_path = f"{ROOT}/{os.getenv('CLASSIFIER_RUNS')}/{run_id}"
        logging.info(f'Loaded run {run_id} checkpoint')
    else:
        logging.info(f'New run: {run_id}')
        run_path = f"{ROOT}/{os.getenv('CLASSIFIER_RUNS')}/{run_id}"
        os.mkdir(run_path)

    logging.info('Starting model evaluation')

    train_loss, train_acc, val_loss, val_acc, best_acc, best_weights = classifier_evaluate(
        model=model, 
        num_epochs=num_epochs, 
        batch_size=batch_size,
        optimizer=optimizer, 
        scheduler=scheduler,
        device=device, 
        train_dataset=train_dataset, 
        val_dataset=val_dataset, 
        criterion=criterion, 
        early_stopper=early_stopper, 
        answer_encoder=answer_encoder,
        max_length=max_length,
        tokenizer=tokenizer,
        feature_extractor=feature_extractor,
        question_encoder=question_encoder,
        train_loss_ckp = train_loss_ckp,
        train_acc_ckp = train_acc_ckp,
        val_loss_ckp = val_loss_ckp,
        val_acc_ckp = val_acc_ckp,
        best_acc_ckp = best_acc_ckp,
        start_epoch = start_epoch,
        run_id=run_id,
        logging=logging)
    
    logging.info(f'Evaluation ended in {len(train_loss)} epochs')
    
    torch.save(best_weights, f"{ROOT}/{run_path}/model.pt")

    with open(f"{ROOT}/{run_path}/run.json", "w") as f:
        config = vars(args)
        config['train_loss'] = train_loss
        config['val_loss'] = val_loss
        config['train_acc'] = train_acc
        config['val_acc'] = val_acc
        config['best_acc'] = best_acc
        config['run_id'] = run_id
        json.dump(config, f)

    os.remove(f"{ROOT}/{os.getenv('CLASSIFIER_CHECKPOINT')}")

    runs = os.listdir(f"{ROOT}/{os.getenv('CLASSIFIER_RUNS')}")
    best_run = None

    if len(runs) > 0:
        runs_best_acc = 0
        for run in runs:
            if os.path.exists(f"{ROOT}/{os.getenv('CLASSIFIER_RUNS')}/{run}/run.json"):
                with open(f"{ROOT}/{os.getenv('CLASSIFIER_RUNS')}/{run}/run.json", 'r') as f:
                    data = json.load(f)
                    runs_best_acc = data['best_acc'] if data['best_acc'] > runs_best_acc else runs_best_acc
                    best_run = run
        if os.path.exists(f"{ROOT}/{os.getenv('CLASSIFIER_MODEL')}"):
            os.remove(f"{ROOT}/{os.getenv('CLASSIFIER_MODEL')}")
        
        shutil.copyfile(f"{ROOT}/{os.getenv('CLASSIFIER_RUNS')}/{run}/model.pt", 
                        f"{ROOT}/{os.getenv('CLASSIFIER_MODEL')}")
        
        shutil.copyfile(f"{ROOT}/{os.getenv('CLASSIFIER_RUNS')}/{run}/run.json", 
                        f"{ROOT}/{os.getenv('CLASSIFIER_RUN')}")
        
        logging.info(f'New best run: {best_run}')

    plot_run(base_path=f"{ROOT}/{os.getenv('CLASSIFIER_RUNS')}", run_id=run_id)

    if turnoff >= 0:
        os.system(f"shutdown /s /t {turnoff}")