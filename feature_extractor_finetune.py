import os
import torch
import json
from dotenv import load_dotenv
from torch import optim
from torch.nn import CrossEntropyLoss
import torch.optim as optim
from datetime import datetime
import shutil
import logging
from callback import EarlyStopper
from util import generate_run_id
from dataset import FeatureExtractor, prepare_feature_extractor_data, df_train_test_split, feature_extractor_class_names
from model import get_feature_extractor_model, feature_extractor_evaluate
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR, LinearLR
import argparse
from plot_generator import plot_run

now = datetime.now()
now = now.strftime("%Y-%m-%d")

logging.basicConfig(
    filename=f"logs/{now}.log",
    encoding="utf-8",
    filemode="a",
    format="{asctime} - {levelname} - {message}",
    style="{",
    datefmt="%Y-%m-%d %H:%M",
    force=True,
    level=logging.INFO
)

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
    parser.add_argument('--model', help="'resnet50', 'resnet101', 'resnet152' or 'vgg16'")
    parser.add_argument('--freeze', help="'0' for inference, '1' for training only the top layer or '2' for training the entire model")
    parser.add_argument('--aug', help="'1' for augmented dataset")

    args = parser.parse_args()

    turnoff = int(args.turnoff)

    freeze = args.freeze or '2'

    model = args.model or 'resnet152'

    run_id = generate_run_id()
    
    use_aug = args.aug == '1'
    
    csv_path = 'FEATURE_EXTRACTOR_CSV_AUG' if use_aug else 'FEATURE_EXTRACTOR_CSV'

    dataset = prepare_feature_extractor_data(csv_path, aug=use_aug)    

    class_names = feature_extractor_class_names()
    
    train_set, val_set = df_train_test_split(dataset, 0.2) 

    logging.info('Building dataloaders...')
    
    train_dataset = FeatureExtractor(
        train_set['path'].to_numpy(), 
        train_set['label'].to_numpy(), 
        train_set['code'].to_numpy(), 
        train_set['bbox'].to_numpy(), 
        train=True)
    
    val_dataset = FeatureExtractor(
        val_set['path'].to_numpy(), 
        val_set['label'].to_numpy(), 
        val_set['code'].to_numpy(), 
        val_set['bbox'].to_numpy(), 
        train=False) 

    num_classes = len(class_names)

    logging.info('Recovering base model...')

    pretrained_model = get_feature_extractor_model(model_name=model, num_classes=num_classes, freeze=freeze)

    if device == 'cuda':
        torch.compile(pretrained_model, 'max-autotune')

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
        optimizer = optim.SGD(pretrained_model.parameters(), lr=lr, momentum=momentum)
    elif args.optimizer == 'adam':
        optimizer = optim.Adam(pretrained_model.parameters(), lr=lr, weight_decay=weight_decay)
    elif args.optimizer == 'adamw':
        optimizer = optim.AdamW(pretrained_model.parameters(), lr=lr, weight_decay=weight_decay)
        
    early_stopper = EarlyStopper(patience=patience, min_delta=min_delta)

    scheduler = None

    if args.scheduler == 'plateau':
        scheduler = ReduceLROnPlateau(optimizer=optimizer, mode=mode)
    elif args.scheduler == 'cosine':
        scheduler = CosineAnnealingLR(optimizer=optimizer, T_max=T_max, eta_min=eta_min)
    elif args.scheduler == 'linear':
        scheduler = LinearLR(optimizer=optimizer)

    pretrained_model.to(device)

    train_loss_ckp = None
    train_acc_ckp = None
    val_loss_ckp = None
    val_acc_ckp = None
    best_acc_ckp = None 
    start_epoch = 1

    run_path = None
    
    if os.path.exists(os.getenv('FEATURE_EXTRACTOR_CHECKPOINT')):
        checkpoint = torch.load(os.getenv('FEATURE_EXTRACTOR_CHECKPOINT'), weights_only=True)
        pretrained_model.load_state_dict(checkpoint['model_state_dict'])
        start_epoch = checkpoint['num_epochs']
        train_loss_ckp = checkpoint['train_loss']
        train_acc_ckp = checkpoint['train_acc']
        val_loss_ckp = checkpoint['val_loss']
        val_acc_ckp = checkpoint['val_acc']
        best_acc_ckp = checkpoint['best_acc']
        run_id = checkpoint['run_id']
        run_path = f"{os.getenv('FEATURE_EXTRACTOR_RUNS')}/{run_id}"
        logging.info(f'Loaded run {run_id} checkpoint')
    else:
        logging.info(f'New run: {run_id}')
        run_path = f"{os.getenv('FEATURE_EXTRACTOR_RUNS')}/{run_id}"
        os.mkdir(run_path)

    logging.info('Starting model evaluation')

    train_loss, train_acc, val_loss, val_acc, best_acc, best_weights = feature_extractor_evaluate(
        model=pretrained_model, 
        num_epochs=num_epochs, 
        batch_size=batch_size,
        optimizer=optimizer, 
        scheduler=scheduler,
        scheduler_name=args.scheduler,
        device=device, 
        train_dataset=train_dataset, 
        val_dataset=val_dataset, 
        criterion=criterion, 
        early_stopper=early_stopper, 
        class_names=class_names,
        train_loss_ckp = train_loss_ckp,
        train_acc_ckp = train_acc_ckp,
        val_loss_ckp = val_loss_ckp,
        val_acc_ckp = val_acc_ckp,
        best_acc_ckp = best_acc_ckp,
        start_epoch = start_epoch,
        run_id=run_id,
        logging=logging)
    
    logging.info(f'Evaluation ended in {len(train_loss)} epochs')
    
    torch.save(best_weights, f"{run_path}/model.pt")

    with open(f"{run_path}/run.json", "w") as f:
        config = vars(args)
        config['train_loss'] = train_loss
        config['val_loss'] = val_loss
        config['train_acc'] = train_acc
        config['val_acc'] = val_acc
        config['best_acc'] = best_acc
        config['run_id'] = run_id
        json.dump(config, f)

    os.remove(os.getenv('FEATURE_EXTRACTOR_CHECKPOINT'))

    runs = os.listdir(f"{os.getenv('FEATURE_EXTRACTOR_RUNS')}")
    best_run = None

    if len(runs) > 0:
        runs_best_acc = 0
        for run in runs:
            if os.path.exists(f"{os.getenv('FEATURE_EXTRACTOR_RUNS')}/{run}/run.json"):
                with open(f"{os.getenv('FEATURE_EXTRACTOR_RUNS')}/{run}/run.json", 'r') as f:
                    data = json.load(f)
                    runs_best_acc = data['best_acc'] if data['best_acc'] > runs_best_acc else runs_best_acc
                    best_run = run
        if os.path.exists(f"{os.getenv('FEATURE_EXTRACTOR_MODEL')}"):
            os.remove(f"{os.getenv('FEATURE_EXTRACTOR_MODEL')}")
        
        shutil.copyfile(f"{os.getenv('FEATURE_EXTRACTOR_RUNS')}/{run}/model.pt", 
                        f"{os.getenv('FEATURE_EXTRACTOR_MODEL')}")
        
        shutil.copyfile(f"{os.getenv('FEATURE_EXTRACTOR_RUNS')}/{run}/run.json", 
                        f"{os.getenv('FEATURE_EXTRACTOR_RUN')}")
        
        logging.info(f'New best run: {best_run}')

    plot_run(base_path=os.getenv('FEATURE_EXTRACTOR_RUNS'), run_id=run_id)

    if turnoff >= 0:
        os.system(f"shutdown /s /t {turnoff}")