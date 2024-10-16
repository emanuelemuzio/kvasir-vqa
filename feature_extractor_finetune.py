import os
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
import torch
import numpy as np
import json
from dotenv import load_dotenv
from torch import optim
from torch.nn import CrossEntropyLoss
import torch.optim as optim
from torcheval.metrics.functional import multiclass_accuracy
from datetime import datetime
import shutil
import logging
from callback import EarlyStopper
from util import generate_run_id
from dataset import Kvasir, prepare_data, df_train_test_split, kvasir_gradcam_class_names, label2id_list
from model import prepare_pretrained_model
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

def evaluate(model, 
            num_epochs, 
            batch_size, 
            optimizer, 
            scheduler,
            scheduler_name,
            device, 
            train_dataset, 
            val_dataset, 
            criterion, 
            early_stopper, 
            train_loss_ckp = [],
            train_acc_ckp = [],
            val_loss_ckp = [],
            val_acc_ckp = [],
            best_acc_ckp = - np.inf,
            start_epoch = 1,
            run_id='test',
            logging=None):
    
    logging.info('Starting model evaluation')

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
    
    train_loss = []
    val_loss = []

    train_acc = []
    val_acc = []
    best_acc = - np.inf
    best_weights = None

    if train_loss_ckp is not None:
        logging.info(f'Loading {run_id} checkpoint')
        train_loss += train_loss_ckp
        train_acc += train_acc_ckp
        val_loss += val_loss_ckp
        val_acc += val_acc_ckp
        best_acc = best_acc_ckp
        early_stopper.min_validation_loss = min(val_loss)

    # Ciclo di addestramento

    for epoch in tqdm(range(start_epoch, num_epochs + 1)):
        torch.cuda.empty_cache()
        epoch_train_loss, epoch_train_acc = train(model, train_dataloader, criterion, optimizer, device, scheduler, scheduler_name)
        train_loss.append(epoch_train_loss)
        train_acc.append(epoch_train_acc)
        
        epoch_val_loss, epoch_val_acc = val(model, val_dataloader, criterion, device, scheduler, scheduler_name)
        val_loss.append(epoch_val_loss)
        val_acc.append(epoch_val_acc)
        
        logging.info(f"Epoch [{epoch}/{num_epochs}] Train Loss: {epoch_train_loss:.4f}  Validation Loss: {epoch_val_loss:.4f}, Validation Accuracy: {epoch_val_acc*100:.2f}%")

        print(f"Epoch [{epoch}/{num_epochs}] Train Loss: {epoch_train_loss:.4f}  Validation Loss: {epoch_val_loss:.4f}, Validation Accuracy: {epoch_val_acc*100:.2f}%")
        
        if epoch_val_acc > best_acc:
            best_acc = epoch_val_acc
            best_weights = model.state_dict()

            torch.save({
                'model_state_dict' : best_weights,
                'num_epochs' : epoch,
                'train_loss' : train_loss,
                'train_acc' : train_acc,
                'val_loss' : val_loss,
                'val_acc' : val_acc,
                'best_acc' : best_acc,
                'run_id' : run_id
            }, os.getenv('KVASIR_GRADCAM_CHECKPOINT')) 

            logging.info('Checkpoint reached')
            
        if early_stopper.early_stop(epoch_val_loss):
            logging.info('Early stop activating')
            break
            
    return train_loss, train_acc, val_loss, val_acc, best_acc, best_weights
        
def train(model, train_dataset, criterion, optimizer, device, scheduler, scheduler_name):
    model.train()

    train_loss = 0.0
    train_acc = 0.0

    for i, (img, label, __, _) in enumerate(train_dataset):

        img = img.to(device)
        label = torch.tensor(label2id_list(label)).to(device)

        optimizer.zero_grad()

        output = model(img)

        loss = criterion(output, label)

        loss.backward()

        optimizer.step()

        acc = multiclass_accuracy(output, label) 
        train_acc += acc.item()
        train_loss += loss.item()

    if scheduler_name == 'cosine':
        scheduler.step()

    train_loss = train_loss / len(train_dataset)
    train_acc = train_acc / len(train_dataset)

    return train_loss, train_acc

def val(model, val_dataset, criterion, device, scheduler, scheduler_name):
    # Evaluate the model on the validation set
    model.eval()
    val_loss = 0.0
    val_acc = 0.0
        
    with torch.no_grad():
        for i, (img, label, _, _) in enumerate(val_dataset):
            img = img.unsqueeze(0).to(device)[0]
            label = torch.tensor(label2id_list(label)).to(device)

            optimizer.zero_grad()

            output = model(img)

            loss = criterion(output, label)
            
            acc = multiclass_accuracy(output, label)
            val_acc += acc.item()
            val_loss += loss.item()

        # Calculate validation loss
        val_loss /= len(val_dataset)

        # Calculate validation accuracy
        val_acc = val_acc / len(val_dataset)
        
        if scheduler_name == 'plateau':
            scheduler.step(val_loss)

        return val_loss, val_acc 
 
if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--turnoff', help="-1 for no turnoff, 0 for instant turnoff, x to plan the turnoff in x seconds") 
    parser.add_argument('--num_epochs')
    parser.add_argument('--batch_size')
    parser.add_argument('--lr')
    parser.add_argument('--momentum')
    parser.add_argument('--T_max')
    parser.add_argument('--eta_min')
    parser.add_argument('--patience')
    parser.add_argument('--min_delta')
    parser.add_argument('--mode')
    parser.add_argument('--scheduler')
    parser.add_argument('--optimizer')
    parser.add_argument('--weight_decay')
    parser.add_argument('--resnet')
    parser.add_argument('--freeze_layers')

    args = parser.parse_args()

    turnoff = int(args.turnoff)

    freeze_layers = args.freeze_layers == '1'

    resnet = args.resnet or os.getenv('RESNET')

    run_id = generate_run_id()

    dataset = prepare_data(os.getenv('KVASIR_GRADCAM_CSV_AUG'), aug=True)    

    class_names = kvasir_gradcam_class_names()
    
    train_set, val_set = df_train_test_split(dataset, 0.2) 
    
    train_dataset = Kvasir(
        train_set['path'].to_numpy(), 
        train_set['label'].to_numpy(), 
        train_set['code'].to_numpy(), 
        train_set['bbox'].to_numpy(), 
        train=True)
    
    val_dataset = Kvasir(
        val_set['path'].to_numpy(), 
        val_set['label'].to_numpy(), 
        val_set['code'].to_numpy(), 
        val_set['bbox'].to_numpy(), 
        train=False) 

    num_classes = len(class_names)

    pretrained_model = prepare_pretrained_model(resnet, num_classes, freeze_layers=freeze_layers, inference=False)

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
    
    if os.path.exists(os.getenv('KVASIR_GRADCAM_CHECKPOINT')):
        checkpoint = torch.load(os.getenv('KVASIR_GRADCAM_CHECKPOINT'), weights_only=True)
        pretrained_model.load_state_dict(checkpoint['model_state_dict'])
        start_epoch = checkpoint['num_epochs']
        train_loss_ckp = checkpoint['train_loss']
        train_acc_ckp = checkpoint['train_acc']
        val_loss_ckp = checkpoint['val_loss']
        val_acc_ckp = checkpoint['val_acc']
        best_acc_ckp = checkpoint['best_acc']
        run_id = checkpoint['run_id']
        run_path = f"{os.getenv('KVASIR_GRADCAM_RUNS')}/{run_id}"
        logging.info(f'Loaded run {run_id} checkpoint')
    else:
        logging.info(f'New run: {run_id}')
        run_path = f"{os.getenv('KVASIR_GRADCAM_RUNS')}/{run_id}"
        os.mkdir(run_path)

    train_loss, train_acc, val_loss, val_acc, best_acc, best_weights = evaluate(
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

    os.remove(os.getenv('KVASIR_GRADCAM_CHECKPOINT'))

    runs = os.listdir(f"{os.getenv('KVASIR_GRADCAM_RUNS')}")
    best_run = None

    if len(runs) > 0:
        runs_best_acc = 0
        for run in runs:
            if os.path.exists(f"{os.getenv('KVASIR_GRADCAM_RUNS')}/{run}/run.json"):
                with open(f"{os.getenv('KVASIR_GRADCAM_RUNS')}/{run}/run.json", 'r') as f:
                    data = json.load(f)
                    runs_best_acc = data['best_acc'] if data['best_acc'] > runs_best_acc else runs_best_acc
                    best_run = run
        if os.path.exists(f"{os.getenv('KVASIR_GRADCAM_MODEL')}"):
            os.remove(f"{os.getenv('KVASIR_GRADCAM_MODEL')}")
        
        shutil.copyfile(f"{os.getenv('KVASIR_GRADCAM_RUNS')}/{run}/model.pt", 
                        f"{os.getenv('KVASIR_GRADCAM_MODEL')}")
        
        shutil.copyfile(f"{os.getenv('KVASIR_GRADCAM_RUNS')}/{run}/run.json", 
                        f"{os.getenv('KVASIR_GRADCAM_RUN')}")
        
        logging.info(f'New best run: {best_run}')

    plot_run(base_path=os.getenv('KVASIR_GRADCAM_RUNS'), run_id=run_id)

    if turnoff >= 0:
        os.system(f"shutdown /s /t {turnoff}")