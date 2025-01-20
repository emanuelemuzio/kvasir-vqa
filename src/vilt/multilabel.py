import sys
sys.path.append('src')

import torch
import os
import pandas as pd
import argparse
import json
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from common.util import ROOT, logger, generate_run_id, plot_run, check_run_type, save_multilabel_results, get_new_tokens
from common.earlystop import EarlyStopper
from vilt.data import MultilabelDataset
from vilt.data import get_multilabel_config as get_config
from dotenv import load_dotenv
from torch import optim
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR, LinearLR, ExponentialLR, StepLR
from sklearn.model_selection import train_test_split
from transformers import ViltProcessor, ViltConfig, ViltForQuestionAnswering
from sklearn.metrics import f1_score

load_dotenv()      
RANDOM_SEED = int(os.getenv('RANDOM_SEED'))
processor = ViltProcessor.from_pretrained("dandelin/vilt-b32-mlm")

def collate_fn(batch):
    input_ids = [item['input_ids'] for item in batch]
    pixel_values = [item['pixel_values'] for item in batch]
    attention_mask = [item['attention_mask'] for item in batch]
    token_type_ids = [item['token_type_ids'] for item in batch]
    labels = [torch.tensor(item['labels']) for item in batch]

    encoding = processor.image_processor.pad(pixel_values, return_tensors="pt")

    batch = {}
    batch['input_ids'] = torch.stack(input_ids)
    batch['attention_mask'] = torch.stack(attention_mask)
    batch['token_type_ids'] = torch.stack(token_type_ids)
    batch['pixel_values'] = encoding['pixel_values']
    batch['pixel_mask'] = encoding['pixel_mask']
    batch['labels'] = torch.stack(labels).float()

    return batch

def evaluate(
            model, 
            num_epochs : int, 
            batch_size : int, 
            optimizer : torch.optim, 
            scheduler : list, 
            device : str, 
            train_dataset : DataLoader, 
            val_dataset : DataLoader, 
            early_stopper : EarlyStopper, 
            train_loss_ckp = [],
            train_acc_ckp = [],
            val_loss_ckp = [],
            val_acc_ckp = [],
            start_epoch = 1,
            run_id='test'): 
    
    logger.info('Starting model evaluation')

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    
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
            }, f"{ROOT}/{os.getenv('VILT_RUNS')}/{run_id}/checkpoint.pt") 

            logger.info('Checkpoint reached')
            
        if early_stopper.early_stop(epoch_val_loss):
            logger.info('Early stop activating')
            break
            
    return train_loss, train_acc, val_loss, val_acc, best_weights

def train(
    model,
    train_dataset, 
    optimizer : torch.optim, 
    device : str): 
    
    model.train()

    train_loss = 0.0
    train_acc = 0.0

    for batch in tqdm(train_dataset, desc='Training phase...'):
        
        optimizer.zero_grad()
        
        data = {k:v.to(device) for k,v in batch.items()}
        
        output = model(**data)

        loss = output.loss
        loss.backward()
        optimizer.step() 
        
        target = data['labels']
        pred = torch.sigmoid(output['logits'])
        pred[pred >= 0.5] = 1
        pred[pred < 0.5] = 0
        
        target = target.cpu().detach().numpy()
        pred = pred.cpu().detach().numpy()
        acc = f1_score(target, pred, average='macro')
        train_acc += acc
        train_loss += loss.item()
        
        if device == 'cuda':
            torch.cuda.empty_cache()

    train_loss = train_loss / len(train_dataset)
    train_acc =train_acc / len(train_dataset)

    return train_loss, train_acc

def val(
    model, 
    val_dataset,  
    scheduler,
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
        for batch in tqdm(val_dataset, desc='Validation phase...'):
        
            data = {k:v.to(device) for k,v in batch.items()}
            
            output = model(**data)

            loss = output.loss
             
            target = data['labels']
            pred = torch.sigmoid(output['logits'])
            pred[pred >= 0.5] = 1
            pred[pred < 0.5] = 0
            
            target = target.cpu().detach().numpy()
            pred = pred.cpu().detach().numpy()
            acc = f1_score(target, pred, average='macro')
            val_acc += acc
            val_loss += loss.item()
            
            del(batch)
            if device == 'cuda':
                torch.cuda.empty_cache()

        val_loss /= len(val_dataset)

        val_acc /= len(val_dataset)
        
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
    targets = []
        
    with torch.no_grad():
        for batch in tqdm(dataset, desc='Test phase...'):
            input_ids = batch['input_ids'].unsqueeze(0)
            pixel_values = batch['pixel_values'].unsqueeze(0)
            attention_mask = batch['attention_mask'].unsqueeze(0)
            token_type_ids = batch['token_type_ids'].unsqueeze(0)
            labels = torch.FloatTensor(batch['labels']).unsqueeze(0)
            
            encoding = processor.image_processor.pad(pixel_values, return_tensors="pt").to(device)
            
            data = {}
            data['input_ids'] = input_ids
            data['attention_mask'] = attention_mask
            data['token_type_ids'] = token_type_ids
            data['pixel_values'] = pixel_values
            data['pixel_mask'] = encoding['pixel_mask']
            data['labels'] = labels
            
            data = {k:v.to(device) for k,v in data.items()}
            
            output = model(**data)

            target = data['labels']
            pred = torch.sigmoid(output['logits'])
            pred[pred >= 0.5] = 1
            pred[pred < 0.5] = 0
            
            target = target.squeeze().cpu().detach().tolist()
            pred = pred.squeeze().cpu().detach().tolist()
            
            predictions.append(pred)
            targets.append(target)
            
            del(batch)
            del(data)
            if device == 'cuda':
                torch.cuda.empty_cache()
                
    return predictions, targets
    
def launch_experiment(args : argparse.Namespace, device: str) -> None:
    
    config = ViltConfig.from_dict(get_config())
    model = ViltForQuestionAnswering.from_pretrained("dandelin/vilt-b32-mlm",
                                                 id2label=config.id2label,
                                                 label2id=config.label2id)
    model.to(device)
    
    new_tokens = get_new_tokens(processor.tokenizer)
    processor.tokenizer.add_tokens(new_tokens)
    model.resize_token_embeddings(len(processor.tokenizer))
    
    print(f"Added {len(new_tokens)} new tokens to tokenizer")
    
    logger.info(f"Launching experiment with configuration: {args}")
    
    kvasir_vqa_datapath = f"{ROOT}/{os.getenv('KVASIR_VQA_DATA')}"
    
    df = pd.read_csv(f"{ROOT}/{os.getenv('METADATA_MULTILABEL_CSV')}") 
    
    logger.info("Dataset retrieved")
        
    df.dropna(axis=0, inplace=True)
    
    x_columns = ['source', 'question', 'img_id']
    y_columns = df.drop(x_columns, axis=1).columns
    
    X = df[x_columns]
    Y = df[y_columns]
    
    logger.info("Splitting data")

    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, 
        test_size=0.3, 
        random_state=RANDOM_SEED, 
        shuffle=True
    )

    X_test, X_val, Y_test, Y_val = train_test_split(
        X_test, Y_test, 
        test_size=0.5,   
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
 
    train_dataset = MultilabelDataset(
        source=X_train['source'].to_numpy(), 
        question=X_train['question'].to_numpy(), 
        answer=Y_train[y_columns].to_numpy(), 
        img_id=X_train['img_id'].to_numpy(), 
        base_path=kvasir_vqa_datapath,
        processor=processor,
        config=config
    )
        
    test_dataset = MultilabelDataset(
        source=X_test['source'].to_numpy(), 
        question=X_test['question'].to_numpy(), 
        answer=Y_test[y_columns].to_numpy(), 
        img_id=X_test['img_id'].to_numpy(),
        base_path=kvasir_vqa_datapath,
        processor=processor,
        config=config
    )
    
    val_dataset = MultilabelDataset(
        source=X_val['source'].to_numpy(), 
        question=X_val['question'].to_numpy(), 
        answer=Y_val[y_columns].to_numpy(), 
        img_id=X_val['img_id'].to_numpy(), 
        base_path=kvasir_vqa_datapath,
        processor=processor,
        config=config
    ) 
    
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
    
    step_size = int(args.step_size) or None
    gamma = float(args.gamma) or None

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
            scheduler.append(ExponentialLR(optimizer=optimizer, gamma=gamma, last_epoch=-1, verbose=False))
        elif name == 'step':
            scheduler.append(StepLR(optimizer, step_size=step_size, gamma=gamma))
    
    '''
    1: Generate new run_id and run folder
    2: Recover from existing checkpoint, finish training and run tests
    3: Use existing run_id, the run folder contains only the model.pt file, run tests for it
    '''
    
    run_type = None
    run_id = None
    delete_ckp = args.delete_ckp == "1"
    
    '''
    <> args.run_id is not empty:
        <> the run_id is valid and the folder exists:
            <> the folder contains a checkpoint.pt file:
                - the training continues -> 2
            <> the folder doesnt contain a checkpoint.pt:
                <> the folder contains a model.pt file:
                    - only run tests -> 3
                <> the folder doesnt contain neither checkpoint nor model:
                    - delete folder, create new one for train and then tests -> 1
        <> the run_id is invalid, the folder doesnt exist:
            - create new folder for train and tests -> 1
    <> args.run_id is empty:
        - generate new id and create new folder -> 1
    '''
    
    if len(args.run_id) > 0:
        if delete_ckp:
            if os.path.exists(f"{ROOT}/{os.getenv('VILT_RUNS')}/{args.run_id}/checkpoint.pt"):
                os.remove(f"{ROOT}/{os.getenv('VILT_RUNS')}/{args.run_id}/checkpoint.pt") 
        run_type = check_run_type(f"{ROOT}/{os.getenv('VILT_RUNS')}/{args.run_id}")
        if run_type != 1:
            run_id = args.run_id
        else:
            logger.info("Generating run id")
            run_id = generate_run_id()
    else:
        logger.info("Generating run id")
        run_id = generate_run_id()
        run_type = 1
                
    train_loss = None
    train_acc = None
    val_loss = None
    val_acc = None
    train_loss_ckp = None
    train_acc_ckp = None
    val_loss_ckp = None
    val_acc_ckp = None
    start_epoch = 1
    best_weights = None     

    run_path = f"{ROOT}/{os.getenv('VILT_RUNS')}/{run_id}"
    
    match run_type:
        case 1:
            logger.info(f'New run: {run_id}')
            os.mkdir(run_path)
    
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
                early_stopper=early_stopper, 
                train_loss_ckp = train_loss_ckp,
                train_acc_ckp = train_acc_ckp,
                val_loss_ckp = val_loss_ckp,
                val_acc_ckp = val_acc_ckp,
                start_epoch = start_epoch,
                run_id=run_id)
                
            logger.info(f'Evaluation ended in {len(train_loss)} epochs')
                
            torch.save(best_weights, f"{run_path}/model.pt")
            
        case 2:
            min_epochs = 0
            checkpoint = torch.load(f"{ROOT}/{os.getenv('VILT_RUNS')}/{run_id}/checkpoint.pt", weights_only=False)
            model.load_state_dict(checkpoint['model_state_dict'])
            start_epoch = checkpoint['num_epochs']
            train_loss_ckp = checkpoint['train_loss']
            train_acc_ckp = checkpoint['train_acc']
            val_loss_ckp = checkpoint['val_loss']
            val_acc_ckp = checkpoint['val_acc']
            run_id = checkpoint['run_id']
            run_path = f"{ROOT}/{os.getenv('VILT_RUNS')}/{run_id}"
            logger.info(f'Loaded run {run_id} checkpoint')
                
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
                early_stopper=early_stopper, 
                train_loss_ckp = train_loss_ckp,
                train_acc_ckp = train_acc_ckp,
                val_loss_ckp = val_loss_ckp,
                val_acc_ckp = val_acc_ckp,
                start_epoch = start_epoch,
                run_id=run_id)
                
            logger.info(f'Evaluation ended in {len(train_loss)} epochs')
                
            torch.save(best_weights, f"{run_path}/model.pt")
            
        case 3:
            best_weights = torch.load(f"{run_path}/model.pt", weights_only=True) 
        
    logger.info(f'Starting test phase')
    
    model.load_state_dict(best_weights)
    
    y_pred, y_true = predict(
        model=model, 
        device=device,
        processor=processor,
        dataset=test_dataset
    )
    
    del(model)
    torch.cuda.empty_cache()    
    
    labels = list(config.id2label.values())
    
    save_multilabel_results(y_true=y_true, y_pred=y_pred, labels=labels, path=run_path, run_id=run_id)
    
    logger.info("Saved test report")
    
    if run_type != 3:
        with open(f"{run_path}/run.json", "w") as f:
            config = vars(args)
            config['train_loss'] = train_loss
            config['val_loss'] = val_loss
            config['train_acc'] = train_acc
            config['val_acc'] = val_acc
            config['run_id'] = run_id
            json.dump(config, f)

        os.remove(f"{run_path}/checkpoint.pt") 

        plot_run(base_path=f"{ROOT}/{os.getenv('VILT_RUNS')}", run_id=run_id)
        
        logger.info("Run plotted and save to run.json file")