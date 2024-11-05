import sys
sys.path.append('src')

from torchvision import models
from dotenv import load_dotenv
import os
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
import torch
from common.util import ROOT, logger
import numpy as np
from dotenv import load_dotenv
from torcheval.metrics.functional import multiclass_accuracy
from common.util import label2id_list
from feature_extractor.data import get_class_names

load_dotenv()  



def get_model(model_name='resnet152', num_classes=0, freeze='2'):
    
    '''
    Function for retrieving the base model for either inference or training.
    The model name is passed from the parameters.
    
    ----------
    Parameters
        model_name: str
            Accepted model names are: resnet (50, 101 and 152) and vgg(16)
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
            best_acc_ckp = - np.inf,
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
    best_acc = - np.inf
    best_weights = None

    if train_loss_ckp is not None:
        logger.info(f'Loading {run_id} checkpoint')
        train_loss += train_loss_ckp
        train_acc += train_acc_ckp
        val_loss += val_loss_ckp
        val_acc += val_acc_ckp
        best_acc = best_acc_ckp
        early_stopper.min_validation_loss = min(val_loss)

    # Ciclo di addestramento

    for epoch in tqdm(range(start_epoch, num_epochs + 1)):
        torch.cuda.empty_cache()
        epoch_train_loss, epoch_train_acc = train(model, train_dataloader, criterion, optimizer, class_names, device, scheduler)
        train_loss.append(epoch_train_loss)
        train_acc.append(epoch_train_acc)
        
        epoch_val_loss, epoch_val_acc = val(model, val_dataloader, criterion, optimizer, class_names, device)
        val_loss.append(epoch_val_loss)
        val_acc.append(epoch_val_acc)
        
        logger.info(f"Epoch [{epoch}/{num_epochs}] Train Loss: {epoch_train_loss:.4f}  Validation Loss: {epoch_val_loss:.4f}, Validation Accuracy: {epoch_val_acc*100:.2f}%")

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
            }, f"{ROOT}/{os.getenv('FEATURE_EXTRACTOR_CHECKPOINT')}") 

            logger.info('Checkpoint reached')
            
        if early_stopper.early_stop(epoch_val_loss):
            logger.info('Early stop activating')
            break
            
    return train_loss, train_acc, val_loss, val_acc, best_acc, best_weights
 
 

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

        img = img.to(device)
        label = (torch.tensor((label2id_list(label, class_names)))).to(device)

        optimizer.zero_grad()

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



def val(model, val_dataset, criterion, optimizer, class_names, device):
    
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
            img = img.unsqueeze(0).to(device)[0]
            label = (torch.tensor((label2id_list(label, class_names)))).to(device)

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
        
        return val_loss, val_acc 
     
     

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
        # feature_extractor = torch.nn.Sequential(*list(model.children())[:-1])
        model.classifier = model.classifier[:-1]
        feature_extractor = model

    return feature_extractor