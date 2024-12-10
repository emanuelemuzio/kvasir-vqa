import sys
sys.path.append('src')

import json
import torch.optim as optim
import argparse
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
import torch.nn as nn
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from common.util import ROOT, logger
from torchvision import models
from torcheval.metrics.functional import multiclass_accuracy
from common.util import label2id_list
from feature_extractor.data import get_class_names
from dotenv import load_dotenv
from sklearn.metrics import classification_report
from common.earlystop import EarlyStopper
from common.util import generate_run_id, ROOT, plot_run, logger
from feature_extractor.data import _Dataset, prepare_data, get_class_names
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR, LinearLR
from torch import optim
from torch.nn import CrossEntropyLoss
from sklearn.model_selection import train_test_split

load_dotenv()  
RANDOM_SEED = int(os.getenv('RANDOM_SEED'))


def get_model(model_name='resnet152', num_classes=0, freeze='2'):
    
    '''
    Function for retrieving the base model for either inference or training.
    The model name is passed from the parameters.
    
    ----------
    Parameters
        model_name: str
            Accepted model names are: resnet (50, 101 and 152), vgg16 and vitb16
        num_classes: int
            number of output classes
        freeze: str
            Inference =  '0', train only top layer = '1', train all layers = '2'
    ----------

    ------
    Return
        model: torchvision.models
            Torch model according to parameters
    ------
    '''
    
    model = None

    if model_name == 'resnet50':
        
        model = models.resnet50()
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    
    elif model_name == 'resnet101':
    
        model = models.resnet101()
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    
    elif model_name == 'resnet152':
    
        model = models.resnet152()
        model.fc = nn.Linear(model.fc.in_features, num_classes)    
    
    elif model_name == 'vgg16':
    
        model = models.vgg16()
        model.classifier[6] = nn.Linear(in_features=4096, out_features=num_classes)

    elif model_name == 'vitb16':
        model = models.vit_b_16()
        model.heads[0] = nn.Linear(768, out_features=num_classes)
        
        

    if model_name.startswith('vgg'):
        if freeze == '0':
            for param in model.parameters():
                param.requires_grad = False
        
        elif freeze == '1':
            for param in model.classifier[0].parameters():
                param.requires_grad = True  
            for param in model.classifier[3].parameters():
                param.requires_grad = True   
        
        elif freeze == '2':
            for param in model.parameters():
                param.requires_grad = True 
                
    elif model_name.startswith('resnet'):
        if freeze == '0':
            for param in model.parameters():
                param.requires_grad = False
        
        elif freeze == '1':
            for param in model.layer4.parameters():
                param.requires_grad = True
        
        elif freeze == '2':
            for param in model.parameters():
                param.requires_grad = True
                
    elif model_name.startswith('vit'):
        if freeze == '0':
            for param in model.parameters():
                param.requires_grad = False
        
        elif freeze == '1':
            for param in model.heads[0].parameters():
                param.requires_grad = True
        
        elif freeze == '2':
            for param in model.parameters():
                param.requires_grad = True

    return model


def evaluate(model, 
            num_epochs, 
            batch_size, 
            optimizer, 
            scheduler,
            device, 
            train_dataset, 
            val_dataset, 
            criterion, 
            early_stopper, 
            class_names=[],
            train_loss_ckp = [],
            train_acc_ckp = [],
            val_loss_ckp = [],
            val_acc_ckp = [],
            start_epoch = 1,
            run_id='test'):
    
    '''
    Evaluation function for the feature extractor, which includes both 
    the train step and the validation step for each epoch.
    
    ----------
    Parameters
        model: torchvision.models
            CNN base model
        num_epochs: int
            max epochs for evaluation
        batch_size: int
            Numerical batch size
        optimizer: torch optimizer
            Torch optimizer (Adam, SGD, AdamW)
        scheduler: torch scheduler/schedulers
            either no scheduler or a list of (cosine annealing, linear, reduce lr on plateau)
        device: str
            'cuda' or 'pc'
        train_dataset/val_dataset: torch Dataset
            Dataset objects, used for creating torch DataLoaders
        criterion: torch loss
            Torch Loss object, in this case Cross Entropy Loss
        early_stopper: EarlyStopper
            EarlyStopper object for interrupting evaluation early 
        class_names: list
            class names set
        *_ckp: list
            in case of interrupted evaluation (recovered from checkpoint), validation and 
            training history will be passed as parameters
        start_epoch: int
            starting epoch (1 or N, depends on when last evaluation was interrupted)
        run_id: str
            Run identifier, the purpose is identifying the correct folder for storing data
    ----------

    ------
    Return
        train_loss: list
            training loss history
        train_acc: list
            training accuracy history
        val_loss: list
            validation loss history
        val_acc:
            validation accuracy history
        best_acc:
            best accuracy achieved
        best_weights:
            dict state of best epoch
    ------
    '''
    
    logger.info('Starting model evaluation')

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
    
    train_loss = []
    val_loss = []

    train_acc = []
    val_acc = []
    best_weights = None

    if train_loss_ckp is not None:
        logger.info(f'Loading {run_id} checkpoint')
        train_loss += train_loss_ckp
        train_acc += train_acc_ckp
        val_loss += val_loss_ckp
        val_acc += val_acc_ckp
        early_stopper.min_validation_loss = min(val_loss)

    # Ciclo di addestramento

    for epoch in tqdm(range(start_epoch, num_epochs + 1)):
        torch.cuda.empty_cache()
        
        epoch_train_loss, epoch_train_acc = train(model, train_dataloader, criterion, optimizer, class_names, device, scheduler)
        train_loss.append(epoch_train_loss)
        train_acc.append(epoch_train_acc)
        
        epoch_val_loss, epoch_val_acc = val(model, val_dataloader, criterion, class_names, device)
        val_loss.append(epoch_val_loss)
        val_acc.append(epoch_val_acc)
        
        logger.info(f"Epoch [{epoch}/{num_epochs}] Train Loss: {epoch_train_loss:.4f}  Validation Loss: {epoch_val_loss:.4f}, Validation Accuracy: {epoch_val_acc*100:.2f}%")

        print(f"Epoch [{epoch}/{num_epochs}] Train Loss: {epoch_train_loss:.4f}  Validation Loss: {epoch_val_loss:.4f}, Validation Accuracy: {epoch_val_acc*100:.2f}%")
        
        if epoch_val_loss < early_stopper.min_validation_loss:
            best_weights = model.state_dict()

            torch.save({
                'model_state_dict' : best_weights,
                'num_epochs' : epoch,
                'train_loss' : train_loss,
                'train_acc' : train_acc,
                'val_loss' : val_loss,
                'val_acc' : val_acc,
                'run_id' : run_id
            }, f"{ROOT}/{os.getenv('FEATURE_EXTRACTOR_CHECKPOINT')}") 

            logger.info('Checkpoint reached')
            
        if early_stopper.early_stop(epoch_val_loss):
            logger.info('Early stop activating')
            break
            
    return train_loss, train_acc, val_loss, val_acc, best_weights
 
 

def train(model, train_dataset, criterion, optimizer, class_names, device, scheduler):
    
    '''
    Train step function
    
    ----------
    Parameters
        model: torchvision.models
            CNN base model
        train_dataset: torch Dataset
            Dataset object
        criterion: torch loss
            Torch Loss object, in this case Cross Entropy Loss
        optimizer: torch optimizer
            Torch optimizer (Adam, SGD, AdamW)
        class_names: list
            class names set
        device: str
            'cuda' or 'pc'
        scheduler: torch scheduler/schedulers
            either no scheduler or a list of (cosine annealing, linear, reduce lr on plateau) 
    ----------

    ------
    Return
        train_loss: list
            training loss history
        train_acc: list
            training accuracy history
    ------
    '''
    
    model.train()

    train_loss = 0.0
    train_acc = 0.0

    for i, (img, label, __, _) in enumerate(train_dataset):
        torch.cuda.empty_cache()
        optimizer.zero_grad()
        
        img = img.to(device)
        label = (torch.tensor((label2id_list(label, class_names)))).to(device)

        output = model(img)

        loss = criterion(output, label)

        loss.backward()

        optimizer.step() 
        
        acc = multiclass_accuracy(output, label) 
        train_acc += acc.item()
        train_loss += loss.item()

    for s in scheduler:
        s.step()

    train_loss = train_loss / len(train_dataset)
    train_acc = train_acc / len(train_dataset)

    return train_loss, train_acc 



def val(model, val_dataset, criterion, class_names, device):
    
    '''
    Validation step function
    
    ----------
    Parameters
        model: torchvision.models
            CNN base model
        val_dataset: torch Dataset
            Dataset object
        criterion: torch loss
            Torch Loss object, in this case Cross Entropy Loss
        optimizer: torch optimizer
            Torch optimizer (Adam, SGD, AdamW)
        class_names: list
            class names set
        device: str
            'cuda' or 'pc'
    ----------

    ------
    Return
        train_loss: list
            training loss history
        train_acc: list
            training accuracy history
    ------
    '''
    
    # Evaluate the model on the validation set
    model.eval()
    val_loss = 0.0
    val_acc = 0.0
        
    with torch.no_grad():
        for i, (img, label, _, _) in enumerate(val_dataset):
            torch.cuda.empty_cache()
            
            img = img.unsqueeze(0).to(device)[0]
            label = (torch.tensor((label2id_list(label, class_names)))).to(device)

            output = model(img)

            loss = criterion(output, label)
            
            acc = multiclass_accuracy(output, label)
            val_acc += acc.item()
            val_loss += loss.item()

        # Calculate validation loss
        val_loss /= len(val_dataset)

        # Calculate validation accuracy
        val_acc = val_acc / len(val_dataset)
        
        return val_loss, val_acc 



def predict(model, device, dataset, class_names):
    dataloader = DataLoader(dataset)
    predictions = []
    targets = []

    with torch.no_grad():
        for i, (img, label, _, _) in enumerate(dataloader):
            torch.cuda.empty_cache()
            
            img = img.to(device)
            label = (torch.tensor((label2id_list(label, class_names))))
            
            output = model(img)
            output = output.squeeze(0).detach().cpu().tolist()
            
            predictions.append(output)
            targets.append(label)

    return predictions, targets
     
     

def init(model_name='resnet152', weights_path=os.getenv('FEATURE_EXTRACTOR_MODEL')):
    
    '''
    Init the feature extractor.
    
    ----------
    Parameters
        model_name: str
            name for model base (clearly it's one of those valid for the get_model function)
        weights_path: str
            path to the .pt file
    ----------

    ------
    Return
        feature_extractor: model
            base model without the last layer for classification
    ------
    '''
    
    class_names = get_class_names()
    num_classes = len(class_names)

    model = get_model(model_name=model_name, num_classes=num_classes)

    model.load_state_dict(torch.load(weights_path))
    
    model.eval()

    feature_extractor = None

    if model_name.startswith('resnet'):
        feature_extractor = torch.nn.Sequential(*list(model.children())[:-1])
    elif model_name.startswith('vgg'):
        model.classifier = model.classifier[:-1]
        feature_extractor = model
    elif model_name.startswith('vit'):
        model.heads = torch.nn.Sequential(torch.nn.Identity())
        feature_extractor = model

    return feature_extractor

def launch_experiment(args : argparse.Namespace, device : str):
    
    freeze = args.freeze or '2'

    model_name = args.model or 'resnet152'

    logger.info('Generating run id')
    run_id = generate_run_id()
    
    use_aug = args.aug == '1'
    
    csv_path = f"{ROOT}/{os.getenv('FEATURE_EXTRACTOR_CSV_AUG')}" if use_aug else f"{ROOT}/{os.getenv('FEATURE_EXTRACTOR_CSV')}"
    
    class_names = get_class_names()
    
    dataset = prepare_data(csv_path, aug=use_aug)  
    
    y_column = 'label'
    x_columns = dataset.columns.to_list()
    x_columns.remove(y_column)
    
    X = dataset.drop(y_column, axis=1)
    Y = dataset.drop(x_columns, axis=1)

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, stratify=Y, random_state=RANDOM_SEED) 
    X_test, X_val, Y_test, Y_val = train_test_split(X_test, Y_test, test_size=0.5, stratify=Y_test, random_state=RANDOM_SEED) 
    
    X_train = X_train.reset_index()
    Y_train = Y_train.reset_index()
    X_test = X_test.reset_index()
    Y_test = Y_test.reset_index()
    X_val = X_val.reset_index()
    Y_val = Y_val.reset_index()

    logger.info('Building dataloaders...')
    
    train_dataset = _Dataset(
        X_train['path'], 
        Y_train['label'], 
        X_train['code'], 
        X_train['bbox']
    )
    
    val_dataset = _Dataset(
        X_test['path'], 
        Y_test['label'], 
        X_test['code'], 
        X_test['bbox']
    ) 
    
    test_dataset = _Dataset(
        X_val['path'], 
        Y_val['label'], 
        X_val['code'], 
        X_val['bbox']
    ) 

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
    
    test_acc = cr['weighted_avg']['f1-score']
    
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
        config['test_accuracy'] = test_acc
        json.dump(config, f)

    os.remove(f"{ROOT}/{os.getenv('FEATURE_EXTRACTOR_CHECKPOINT')}")

    plot_run(base_path=f"{ROOT}/{os.getenv('FEATURE_EXTRACTOR_RUNS')}", run_id=run_id)