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
from sklearn.preprocessing import LabelEncoder
from torcheval.metrics.functional import multiclass_accuracy
from datetime import datetime
import shutil
import logging
from callback import EarlyStopper
from util import generate_run_id
from dataset import Kvasir, prepare_data, df_train_test_split
from model import prepare_pretrained_model

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
            device, 
            train_dataset, 
            val_dataset, 
            criterion, 
            early_stopper, 
            encoder, 
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
        epoch_train_loss, epoch_train_acc = train(model, train_dataloader, criterion, optimizer, device, encoder)
        train_loss.append(epoch_train_loss)
        train_acc.append(epoch_train_acc)
        
        epoch_val_loss, epoch_val_acc = val(model, val_dataloader, criterion, device, encoder)
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
        
def train(model, train_dataset, criterion, optimizer, device, encoder):
    model.train()

    train_loss = 0.0
    train_acc = 0.0

    for i, (img, label, __, _) in enumerate(train_dataset):

        img = img.unsqueeze(0).to(device)[0]
        label = torch.tensor(encoder.transform(label)).to(device)

        optimizer.zero_grad()

        output = model(img)

        loss = criterion(output, label)

        loss.backward()

        optimizer.step()

        acc = multiclass_accuracy(output, label) 
        train_acc += acc.item()
        train_loss += loss.item()

    train_loss = train_loss / len(train_dataset)
    train_acc = train_acc / len(train_dataset)

    return train_loss, train_acc

def val(model, val_dataset, criterion, device, encoder):
    # Evaluate the model on the validation set
    model.eval()
    val_loss = 0.0
    val_acc = 0.0
        
    with torch.no_grad():
        for i, (img, label, _, _) in enumerate(val_dataset):
            img = img.unsqueeze(0).to(device)[0]
            label = torch.tensor(encoder.transform(label)).to(device)

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
 
if __name__ == '__main__':
    run_id = generate_run_id()

    dataset = prepare_data()    

    class_names = dataset.label.unique()
    
    enc = LabelEncoder()
    enc.fit(class_names)
    
    train_set, val_set = df_train_test_split(dataset, 0.2)
    
    train_dataset = Kvasir(
        train_set['path'].to_numpy(), 
        train_set['label'].to_numpy(), 
        train_set['code'].to_numpy(), 
        train_set['bbox'].to_numpy(), 
        train=True)
    
    val_dataset = Kvasir(val_set['path'].to_numpy(), 
                         val_set['label'].to_numpy(), 
                         val_set['code'].to_numpy(), 
                         val_set['bbox'].to_numpy(), 
                         train=False)

    num_classes = len(class_names)

    pretrained_model = prepare_pretrained_model(num_classes)

    logging.info('Declaring hyper parameters')
    
    # # Train run hyper parameters
    # Define loss function (Binary Cross Entropy Loss in this case, for multi-label classification)

    criterion = CrossEntropyLoss()
    lr = 0.002
    momentum = 0.9
    # Define optimizer
    optimizer = optim.SGD(pretrained_model.parameters(), lr=lr, momentum=momentum)
    early_stopper = EarlyStopper(patience=10, min_delta=0.03)
    # Training loop
    num_epochs = 500
    batch_size = 16
    
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
        device=device, 
        train_dataset=train_dataset, 
        val_dataset=val_dataset, 
        criterion=criterion, 
        early_stopper=early_stopper, 
        encoder=enc,
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
        config = {
            'criterion' : criterion.__class__.__name__,
            'lr' : lr,
            'num_epochs' : num_epochs,
            'batch_size' : batch_size,
            'momentum' : momentum,
            'train_loss' : train_loss,
            'train_acc' : train_acc,
            'val_loss' : val_loss,
            'val_acc' : val_acc,
            'best_acc' : best_acc,
            'run_id' : run_id
        }
        json.dump(config, f)

    os.remove(os.getenv('KVASIR_GRADCAM_CHECKPOINT'))

    runs = os.listdir(f"{os.getenv('KVASIR_GRADCAM_RUNS')}")
    best_run = None

    if len(runs) > 0:
        runs_best_acc = 0
        for run in runs:
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