import sys
sys.path.append('src')

import warnings
warnings.filterwarnings('ignore')

import os
import torch
import json
import torch.optim as optim
import argparse
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from dotenv import load_dotenv
from sklearn.metrics import classification_report
from common.earlystop import EarlyStopper
from common.util import generate_run_id, ROOT, plot_run, logger, df_train_test_split
from feature_extractor.data import _Dataset, prepare_data, get_class_names
from feature_extractor.model import get_model, evaluate, predict
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR, LinearLR
from torch import optim
from torch.nn import CrossEntropyLoss

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
    parser.add_argument('--scheduler', help="'cosine', 'plateau' and 'linear'. write as a csv row for multiple schedulers")
    parser.add_argument('--optimizer', help="'sgd', 'adam' or 'adamw")
    parser.add_argument('--weight_decay', help="1e-2 or 1e-3")
    parser.add_argument('--model', help="'resnet50', 'resnet101', 'resnet152', 'vgg16' or 'vit16b")
    parser.add_argument('--freeze', help="'0' for inference, '1' for training only the top layer or '2' for training the entire model")
    parser.add_argument('--aug', help="'1' for augmented dataset")

    args = parser.parse_args()

    turnoff = int(args.turnoff)

    freeze = args.freeze or '2'

    model_name = args.model or 'resnet152'

    logger.info('Generating run id')
    run_id = generate_run_id()
    
    use_aug = args.aug == '1'
    
    csv_path = f"{ROOT}/{os.getenv('FEATURE_EXTRACTOR_CSV_AUG')}" if use_aug else f"{ROOT}/{os.getenv('FEATURE_EXTRACTOR_CSV')}"
    
    dataset = prepare_data(csv_path, aug=use_aug)    

    class_names = get_class_names()
    
    train_set, val_set = df_train_test_split(dataset, 0.3) 
    val_set, test_set = df_train_test_split(dataset, 0.5) 

    logger.info('Building dataloaders...')
    
    train_dataset = _Dataset(
        train_set['path'].to_numpy(), 
        train_set['label'].to_numpy(), 
        train_set['code'].to_numpy(), 
        train_set['bbox'].to_numpy())
    
    val_dataset = _Dataset(
        val_set['path'].to_numpy(), 
        val_set['label'].to_numpy(), 
        val_set['code'].to_numpy(), 
        val_set['bbox'].to_numpy()) 
    
    test_dataset = _Dataset(
        test_set['path'].to_numpy(), 
        test_set['label'].to_numpy(), 
        test_set['code'].to_numpy(), 
        test_set['bbox'].to_numpy()) 

    num_classes = len(class_names)

    logger.info('Recovering base model...')

    model = get_model(model_name=model_name, num_classes=num_classes, freeze=freeze).to(device)

    if device == 'cuda':
        torch.compile(model, 'max-autotune')

    logger.info('Declaring hyper parameters')
    
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

    train_loss_ckp = None
    train_acc_ckp = None
    val_loss_ckp = None
    val_acc_ckp = None
    start_epoch = 1

    run_path = None
    
    if os.path.exists(f"{ROOT}/{os.getenv('FEATURE_EXTRACTOR_CHECKPOINT')}"):
        checkpoint = torch.load(f"{ROOT}/{os.getenv('FEATURE_EXTRACTOR_CHECKPOINT')}", weights_only=True)
        model.load_state_dict(checkpoint['model_state_dict'])
        start_epoch = checkpoint['num_epochs']
        train_loss_ckp = checkpoint['train_loss']
        train_acc_ckp = checkpoint['train_acc']
        val_loss_ckp = checkpoint['val_loss']
        val_acc_ckp = checkpoint['val_acc']
        run_id = checkpoint['run_id']
        run_path = f"{ROOT}/{os.getenv('FEATURE_EXTRACTOR_RUNS')}/{run_id}"
        logger.info(f'Loaded run {run_id} checkpoint')
    else:
        logger.info(f'New run: {run_id}')
        run_path = f"{ROOT}/{os.getenv('FEATURE_EXTRACTOR_RUNS')}/{run_id}"
        os.mkdir(run_path)
        
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

    logger.info('Starting model evaluation')

    train_loss, train_acc, val_loss, val_acc, best_weights = evaluate(
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
        class_names=class_names,
        train_loss_ckp = train_loss_ckp,
        train_acc_ckp = train_acc_ckp,
        val_loss_ckp = val_loss_ckp,
        val_acc_ckp = val_acc_ckp,
        start_epoch = start_epoch,
        run_id=run_id)
    
    logger.info(f'Evaluation ended in {len(train_loss)} epochs')
    
    torch.save(best_weights, f"{run_path}/model.pt")
    
    logger.info(f'Starting test phase')
    
    model.load_state_dict(best_weights)

    predictions, target = predict(
        model=model, 
        device=device, 
        dataset=test_dataset,
        class_names=class_names 
    )
    
    test_acc = None
    
    y_true = []
    y_pred = []
    
    for pred, t in zip(predictions, target):
        y_pred.append(int(np.argmax(pred)))
        y_true.append(t.item())
        
    fig, ax = plt.subplots(nrows=1, ncols=1)
        
    cr = classification_report(y_true=y_true, y_pred=y_pred, target_names=class_names, output_dict=True)
    
    cr = pd.DataFrame(cr).iloc[:-1, :].T
    
    cr.to_csv(f"{run_path}/cr.csv")
    
    sns.heatmap(cr, annot=True, ax=ax).get_figure()
    
    fig.savefig(f"{run_path}/cr.png")
    
    plt.close()

    with open(f"{run_path}/run.json", "w") as f:
        config = vars(args)
        config['train_loss'] = train_loss
        config['val_loss'] = val_loss
        config['train_acc'] = train_acc
        config['val_acc'] = val_acc
        config['run_id'] = run_id
        json.dump(config, f)

    os.remove(f"{ROOT}/{os.getenv('FEATURE_EXTRACTOR_CHECKPOINT')}")

    runs = os.listdir(f"{ROOT}/{os.getenv('FEATURE_EXTRACTOR_RUNS')}") 

    plot_run(base_path=f"{ROOT}/{os.getenv('FEATURE_EXTRACTOR_RUNS')}", run_id=run_id)

    if turnoff >= 0:
        os.system(f"shutdown /s /t {turnoff}")