import sys
sys.path.append('src')

import torch
import os
import numpy as np
import pandas as pd
import argparse
import seaborn as sns
import json
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from common.util import ROOT, logger, generate_run_id, plot_run
from common.earlystop import EarlyStopper
from blip.data import Dataset_
from dotenv import load_dotenv
from sklearn.metrics import classification_report
from torch import optim
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR, LinearLR, ExponentialLR, StepLR
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
from transformers import BlipProcessor, BlipForQuestionAnswering

load_dotenv()      
RANDOM_SEED = int(os.getenv('RANDOM_SEED'))

def evaluate(
            model, 
            processor,
            num_epochs : int, 
            batch_size : int, 
            optimizer : torch.optim, 
            scheduler : list, 
            device : str, 
            scaler,
            train_dataset : DataLoader, 
            val_dataset : DataLoader, 
            early_stopper : EarlyStopper, 
            train_loss_ckp = [],
            train_acc_ckp = [],
            val_loss_ckp = [],
            val_acc_ckp = [],
            start_epoch = 1,
            run_id='test'):
    
    '''
    Evaluation function for the classifier, which includes both 
    the train step and the validation step for each epoch.
    
    ----------
    Parameters
        model: Classifier
            Classifier model
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
        early_stopper: EarlyStopper
            EarlyStopper object for interrupting evaluation early 
        answer_encoder: LabelEncoder
            LabelEncoder fitted on the answers contained in the metadata.csv
        max_length: int
            Max tokenizer length
        tokenizer: AutoTokenizer    
            HuggingFace AutoTokenizer
        feature_extractor: 
            Base Net used for the feature extraction
        question_encoder: AutoModel
            HuggingFace AutoModel for word embeddings
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

        epoch_train_loss, epoch_train_acc = train(
            model,
            scaler,
            train_dataloader, 
            optimizer, 
            device
        )
        
        train_loss.append(epoch_train_loss)
        train_acc.append(epoch_train_acc)
        
        epoch_val_loss, epoch_val_acc = val(
            model, 
            val_dataloader,
            scheduler, 
            device
        )
        
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
            }, f"{ROOT}/{os.getenv('BLIP_CHECKPOINT')}") 

            logger.info('Checkpoint reached')
            
        if early_stopper.early_stop(epoch_val_loss):
            logger.info('Early stop activating')
            break
            
    return train_loss, train_acc, val_loss, val_acc, best_weights



def train(
    model,
    scaler,
    train_dataset : Dataset_, 
    optimizer : torch.optim, 
    device : str):
    
    '''
    Train step function
    
    ----------
    Parameters
        model: Classifier
            Classifier model
        train_dataset: Dataset
            Dataset object
        optimizer: torch optimizer
            Torch optimizer (Adam, SGD, AdamW)
        answer_encoder: LabelEncoder
            LabelEncoder fitted on metadata.csv answers
        max_length: int
            Tokenizer max length
        tokenizer: AutoTokenizer
            HuggingFace Tokenizer
        question_encoder: AutoModel 
            HuggingFace Model for word embeddings
        feature_extractor: model
            Sliced model for feature extraction
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

    for idx, batch in zip(tqdm(range(len(train_dataset)), desc='Training batch: ...'), train_dataset):
        
        input_ids = batch.pop('input_ids').to(device)
        pixel_values = batch.pop('pixel_values').to(device)
        attention_masked = batch.pop('attention_mask').to(device)
        labels = batch.pop('labels').to(device)
        
        with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
            outputs = model(input_ids=input_ids,
                        pixel_values=pixel_values,
                        # attention_mask=attention_masked,
                        labels=labels)
            
        loss = outputs.loss
        train_loss += loss.item()
        # loss.backward()
        # optimizer.step()
        optimizer.zero_grad()
        
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        del(batch)
        if device == 'cuda':
            torch.cuda.empty_cache()

    train_loss = train_loss / len(train_dataset)
    train_acc = train_acc / len(train_dataset)

    return train_loss, train_acc

def val(
    model, 
    val_dataset : Dataset_,  
    scheduler,
    processor,
    device : str):
    
    '''
    Validation step function
    
    ----------
    Parameters
        model: Classifier
            Classifier model
        val_dataset: torch Dataset
            Dataset object
        optimizer: torch optimizer
            Torch optimizer (Adam, SGD, AdamW)
        answer_encoder: LabelEncoder
            LabelEncoder fitted on metadata.csv answers
        max_length: int
            Tokenizer max length
        question_encoder: AutoModel 
            HuggingFace Model for word embeddings
        tokenizer: AutoTokenizer
            HuggingFace Tokenizer
        feature_extractor: model
            Sliced model for feature extraction
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
        for idx, batch in zip(range(len(val_dataset)), val_dataset):
            
            input_ids = batch.pop('input_ids').to(device)
            pixel_values = batch.pop('pixel_values').to(device)
            attention_masked = batch.pop('attention_mask').to(device)
            labels = batch.pop('labels').to(device)

            with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
                outputs = model(input_ids=input_ids,
                            pixel_values=pixel_values,
                            attention_mask=attention_masked,
                            labels=labels)
                
                loss = outputs.loss
                val_loss += loss.item()
            
                del(batch)
                if device == 'cuda':
                    torch.cuda.empty_cache()

        # Calculate validation loss
        val_loss /= len(val_dataset)

        # Calculate validation accuracy
        val_acc = val_acc / len(val_dataset) 
        
        for s in scheduler:
            s.step(val_loss)

        return val_loss, val_acc 
    
def predict(
    model, 
    device,
    processor,
    dataset
    ):
    
    model.eval()
    
    predictions = []
    target = []
        
    with torch.no_grad():
        for item in dataset:
            input_ids = item['input_ids']
            pixel_values = item['pixel_values']
            attention_mask = item['attention_mask']
            token_type_ids = item['token_type_ids']
            labels = item['labels']
            
            encoding = processor.image_processor.pad(pixel_values, return_tensors="pt").to(device)
            
            data = {}
            data['input_ids'] = torch.stack(input_ids)
            data['attention_mask'] = torch.stack(attention_mask)
            data['token_type_ids'] = torch.stack(token_type_ids)
            data['pixel_values'] = encoding['pixel_values']
            data['pixel_mask'] = encoding['pixel_mask']
            data['labels'] = torch.stack(labels)
            
            data = {k:v.to(device) for k,v in data.items()}
            
            output = model(**data)

            targets = torch.tensor([item for item in batch['target']]).to(device)
            
            del(batch)
            if device == 'cuda':
                torch.cuda.empty_cache()
                
    return predictions, target
    
def launch_experiment(args : argparse.Namespace, device: str) -> None:
    
    model = BlipForQuestionAnswering.from_pretrained("Salesforce/blip-vqa-base")
    processor = BlipProcessor.from_pretrained("Salesforce/blip-vqa-base")
    model.to(device)
    
    logger.info(f"Launching experiment with configuration: {args}")
    
    prompting = args.prompting == '1'
    use_aug = args.use_aug == '1'
    
    logger.info("Generating run id")
  
    run_id = generate_run_id() 
    
    df = None
    
    kvasir_vqa_datapath = f"{ROOT}/{os.getenv('KVASIR_VQA_DATA')}"
    kvasir_vqa_datapath_aug = f"{ROOT}/{os.getenv('KVASIR_VQA_DATA_AUG')}"
    
    if prompting:
        if use_aug:
            df = pd.read_csv(f"{ROOT}/{os.getenv('KVASIR_VQA_PROMPT_CSV_AUG')}")
        else:
            df = pd.read_csv(f"{ROOT}/{os.getenv('KVASIR_VQA_PROMPT_CSV')}")
    else:
        if use_aug:
            df = pd.read_csv(f"{ROOT}/{os.getenv('KVASIR_VQA_CSV_AUG')}")
        else:
            df = pd.read_csv(f"{ROOT}/{os.getenv('KVASIR_VQA_CSV')}") 
    
    logger.info("Dataset retrieved")
        
    df.dropna(axis=0, inplace=True)
    
    y_column = 'answer'
    x_columns = df.columns.to_list()
    x_columns.remove(y_column)
    
    X = df.drop(y_column, axis=1)
    Y = df[y_column]
    
    logger.info("Splitting data")

    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, 
        test_size=0.4, 
        stratify=Y, 
        random_state=RANDOM_SEED, 
        shuffle=True
    )

    X_test, X_val, Y_test, Y_val = train_test_split(
        X_test, Y_test, 
        test_size=0.5,   
        stratify=Y_test, 
        random_state=RANDOM_SEED, 
        shuffle=True
    )
    
    X_train = X_train.reset_index()
    Y_train = Y_train.reset_index()
    X_test = X_test.reset_index()
    Y_test = Y_test.reset_index()
    X_val = X_val.reset_index()
    Y_val = Y_val.reset_index()
    
    logger.info('Building dataloaders...')
 
    train_dataset = Dataset_(
        source=X_train['source'].to_numpy(), 
        question=X_train['question'].to_numpy(), 
        answer=Y_train['answer'].to_numpy(), 
        img_id=X_train['img_id'].to_numpy(), 
        base_path=kvasir_vqa_datapath,
        aug_path=kvasir_vqa_datapath_aug,
        processor=processor,
        config=config
    )
        
    test_dataset = Dataset_(
        source=X_test['source'].to_numpy(), 
        question=X_test['question'].to_numpy(), 
        answer=Y_test['answer'].to_numpy(), 
        img_id=X_test['img_id'].to_numpy(),
        base_path=kvasir_vqa_datapath,
        aug_path=kvasir_vqa_datapath_aug,
        processor=processor,
        config=config
    )
    
    val_dataset = Dataset_(
        source=X_val['source'].to_numpy(), 
        question=X_val['question'].to_numpy(), 
        answer=Y_val['answer'].to_numpy(), 
        img_id=X_val['img_id'].to_numpy(), 
        base_path=kvasir_vqa_datapath,
        aug_path=kvasir_vqa_datapath_aug,
        processor=processor,
        config=config
    )
    
    if prompting: 
        train_dataset.add_prompts(X_train['prompt'])
        test_dataset.add_prompts(X_test['prompt'])
        val_dataset.add_prompts(X_val['prompt'])
    
    if device == 'cuda':
        torch.compile(model, 'max-autotune')

    logger.info('Declaring hyper parameters')
    
    # Train run hyper parameters

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
    
    if os.path.exists(f"{ROOT}/{os.getenv('BLIP_CHECKPOINT')}"):
        min_epochs = 0
        checkpoint = torch.load(f"{ROOT}/{os.getenv('BLIP_CHECKPOINT')}", weights_only=True)
        model.load_state_dict(checkpoint['model_state_dict'])
        start_epoch = checkpoint['num_epochs']
        train_loss_ckp = checkpoint['train_loss']
        train_acc_ckp = checkpoint['train_acc']
        val_loss_ckp = checkpoint['val_loss']
        val_acc_ckp = checkpoint['val_acc']
        run_id = checkpoint['run_id']
        run_path = f"{ROOT}/{os.getenv('BLIP_RUNS')}/{run_id}"
        logger.info(f'Loaded run {run_id} checkpoint')
    else:
        logger.info(f'New run: {run_id}')
        run_path = f"{ROOT}/{os.getenv('BLIP_RUNS')}/{run_id}"
        os.mkdir(run_path)
        
    # Define optimizer, scheduler and early stop

    optimizer = None

    if args.optimizer == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum)
    elif args.optimizer == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif args.optimizer == 'adamw':
        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        
    min_epochs = int(args.min_epochs) or 0
        
    early_stopper = EarlyStopper(patience=patience, min_delta=min_delta, min_epochs=min_epochs)

    scheduler = []
    
    scheduler_names = args.scheduler.split(',')

    for name in scheduler_names:
        if name == 'plateau':
            scheduler.append(ReduceLROnPlateau(optimizer=optimizer, mode=mode))
        elif name == 'cosine':
            scheduler.append(CosineAnnealingLR(optimizer=optimizer, T_max=T_max, eta_min=eta_min))
        elif name == 'linear':
            scheduler.append(LinearLR(optimizer=optimizer))
        elif name == 'exponential':
            scheduler.append(ExponentialLR(optimizer=optimizer, gamma=0.9, last_epoch=-1, verbose=False))
        elif name == 'step':
            scheduler.append(StepLR(optimizer, step_size=15, gamma=0.1))

    test_run = args.test_run or None
    
    scaler = torch.amp.GradScaler()
    
    train_loss = None
    train_acc = None
    val_loss = None
    val_acc = None
    best_weights = None
    
    if test_run is not None:
        run_id = test_run
        run_path = f"{ROOT}/{os.getenv('BLIP_RUNS')}/{run_id}"
        run = torch.load(f"{run_path}/model.pt", weights_only=True)
        best_weights = run['model_state_dict']
        train_loss = run['train_loss']
        train_acc = run['train_acc']
        val_loss = run['val_loss']
        val_acc = run['val_acc']
        logger.info(f'Loaded data for testing run {run_id}')
    else:
    
        logger.info('Starting model evaluation')
    
        train_loss, train_acc, val_loss, val_acc, best_weights = evaluate(
            model=model, 
            processor=processor,
            num_epochs=num_epochs, 
            batch_size=batch_size,
            optimizer=optimizer, 
            scheduler=scheduler,
            device=device, 
            scaler=scaler,
            train_dataset=train_dataset, 
            val_dataset=val_dataset, 
            early_stopper=early_stopper, 
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
        dataset=test_dataset
    )
    
    test_acc = None
    
    y_true = []
    y_pred = []
    
    for pred, t in zip(predictions, target):
        y_pred.append(int(np.argmax(pred)))
        y_true.append(t.item())
        
    fig, ax = plt.subplots(nrows=1, ncols=1)
        
    cr = classification_report(y_true=y_true, y_pred=y_pred, target_names=config.id2label.keys(), output_dict=True)
    
    test_acc = cr['macro avg']['f1-score']
    
    cr = pd.DataFrame(cr).iloc[:-1, :].T
    
    cr.to_csv(f"{run_path}/cr.csv")
    
    logger.info("Saved test report")
    
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
        config['test_acc'] = test_acc
        json.dump(config, f)

    os.remove(f"{ROOT}/{os.getenv('BLIP_CHECKPOINT')}") 

    plot_run(base_path=f"{ROOT}/{os.getenv('BLIP_RUNS')}", run_id=run_id)
    
    logger.info("Run plotted and save to run.json file")