import sys
sys.path.append('src')

import torch
import os
import pandas as pd
import argparse
import json
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from common.util import ROOT, logger, generate_run_id, generative_report, check_run_type
from common.earlystop import EarlyStopper
from blip.data import Dataset_
from dotenv import load_dotenv
from torch import optim
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR, LinearLR, ExponentialLR, StepLR
from sklearn.model_selection import train_test_split
from transformers import BlipProcessor, BlipForQuestionAnswering

load_dotenv()      
RANDOM_SEED = int(os.getenv('RANDOM_SEED'))

def evaluate(
            model, 
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
            val_loss_ckp = [],
            start_epoch = 1,
            run_id='test'):
    
    logger.info('Starting model evaluation')

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
    
    train_loss = []
    val_loss = []

    best_weights = None

    if train_loss_ckp is not None:
        logger.info(f'Loading {run_id} checkpoint')
        train_loss += train_loss_ckp
        val_loss += val_loss_ckp
        early_stopper.min_validation_loss = min(val_loss)

    # Ciclo di addestramento

    for epoch in tqdm(range(start_epoch, num_epochs + 1)):
        torch.cuda.empty_cache()

        epoch_train_loss = train(
            model,
            scaler,
            train_dataloader, 
            optimizer, 
            device
        )
        
        train_loss.append(epoch_train_loss)
        
        epoch_val_loss = val(
            model, 
            val_dataloader,
            scheduler, 
            device
        )
        
        val_loss.append(epoch_val_loss)
        
        logger.info(f"Epoch [{epoch}/{num_epochs}] Validation Loss: {epoch_val_loss:.4f}")

        print(f"Epoch [{epoch}/{num_epochs}] Validation Loss: {epoch_val_loss:.4f}")
        
        if epoch_val_loss < early_stopper.min_validation_loss:
            best_weights = model.state_dict()

            torch.save({
                'model_state_dict' : best_weights,
                'num_epochs' : epoch,
                'train_loss' : train_loss,
                'val_loss' : val_loss,
                'run_id' : run_id
            }, f"{ROOT}/{os.getenv('BLIP_CHECKPOINT')}") 

            logger.info('Checkpoint reached')
            
        if early_stopper.early_stop(epoch_val_loss):
            logger.info('Early stop activating')
            break
            
    return train_loss, val_loss, best_weights



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
    ------
    '''
    
    model.train()

    train_loss = 0.0

    for idx, batch in zip(range(len(train_dataset)), train_dataset):
        
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

    return train_loss

def val(
    model, 
    val_dataset : Dataset_,  
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
    ------
    '''
    
    # Evaluate the model on the validation set
    model.eval()
    val_loss = 0.0
        
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

        for s in scheduler:
            s.step(val_loss)

        return val_loss 
    
def predict(
    model, 
    device,
    dataset
    ):
    
    model.eval()
    
    candidate_list = []
    reference_list = []
        
    with torch.no_grad():
        for item in dataset:
            
            input_ids = item['input_ids'].to(device)
            pixel_values = item['pixel_values'].to(device)
            attention_masked = item['attention_mask'].to(device)
            labels = item['labels'].to(device)

            outputs = model.generate(
                input_ids=input_ids,
                pixel_values=pixel_values,
                attention_mask=attention_masked,
                labels=labels
            )
            
            generated_text = dataset.processor.decode(outputs[0], skip_special_tokens=True)
                
            del(batch)
            if device == 'cuda':
                torch.cuda.empty_cache()
                
    return candidate_list, reference_list
    
def launch_experiment(args : argparse.Namespace, device: str) -> None:
    
    model = BlipForQuestionAnswering.from_pretrained("Salesforce/blip-vqa-base")
    processor = BlipProcessor.from_pretrained("Salesforce/blip-vqa-base")
    model.to(device)
    
    logger.info(f"Launching experiment with configuration: {args}")
    
    prompting = args.prompting == '1'
    use_aug = args.use_aug == '1'
    
    logger.info("Generating run id")
  
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
    )
        
    test_dataset = Dataset_(
        source=X_test['source'].to_numpy(), 
        question=X_test['question'].to_numpy(), 
        answer=Y_test['answer'].to_numpy(), 
        img_id=X_test['img_id'].to_numpy(),
        base_path=kvasir_vqa_datapath,
        aug_path=kvasir_vqa_datapath_aug,
        processor=processor,
    )
    
    val_dataset = Dataset_(
        source=X_val['source'].to_numpy(), 
        question=X_val['question'].to_numpy(), 
        answer=Y_val['answer'].to_numpy(), 
        img_id=X_val['img_id'].to_numpy(), 
        base_path=kvasir_vqa_datapath,
        aug_path=kvasir_vqa_datapath_aug,
        processor=processor,
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
    
    scaler = torch.amp.GradScaler(device)
                
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

    step_size = int(args.step_size) or None
    gamma = float(args.gamma) or None

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
            if os.path.exists(f"{ROOT}/{os.getenv('BLIP_RUNS')}/{args.run_id}/checkpoint.pt"):
                os.remove(f"{ROOT}/{os.getenv('BLIP_RUNS')}/{args.run_id}/checkpoint.pt") 
        run_type = check_run_type(f"{ROOT}/{os.getenv('BLIP_RUNS')}/{args.run_id}")
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
    val_loss = None
    train_loss_ckp = None
    val_loss_ckp = None
    start_epoch = 1
    best_weights = None     

    run_path = f"{ROOT}/{os.getenv('BLIP_RUNS')}/{run_id}"
    
    match run_type:
        case 1:
            logger.info(f'New run: {run_id}')
            os.mkdir(run_path)
    
            logger.info('Starting model evaluation')
            
            train_loss, val_loss, best_weights = evaluate(
                model=model, 
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
                val_loss_ckp = val_loss_ckp,
                start_epoch = start_epoch,
                run_id=run_id)
                
            logger.info(f'Evaluation ended in {len(train_loss)} epochs')
                
            torch.save(best_weights, f"{run_path}/model.pt")
            
        case 2:
            min_epochs = 0
            checkpoint = torch.load(f"{ROOT}/{os.getenv('BLIP_RUNS')}/{run_id}/checkpoint.pt", weights_only=False)
            model.load_state_dict(checkpoint['model_state_dict'])
            start_epoch = checkpoint['num_epochs']
            train_loss_ckp = checkpoint['train_loss']
            val_loss_ckp = checkpoint['val_loss']
            run_id = checkpoint['run_id']
            run_path = f"{ROOT}/{os.getenv('BLIP_RUNS')}/{run_id}"
            logger.info(f'Loaded run {run_id} checkpoint')
                
            logger.info('Starting model evaluation')
        
            train_loss, val_loss, best_weights = evaluate(
               model=model, 
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
                val_loss_ckp = val_loss_ckp,
                start_epoch = start_epoch,
                run_id=run_id)
                
            logger.info(f'Evaluation ended in {len(train_loss)} epochs')
                
            torch.save(best_weights, f"{run_path}/model.pt")
            
        case 3:
    
            val_loss = checkpoint['val_loss']
            train_loss = checkpoint['train_loss']
            best_weights = torch.load(f"{run_path}/model.pt", weights_only=True) 
        
    logger.info(f'Starting test phase')
    
    model.load_state_dict(best_weights)
    
    candidate_list, reference_list = predict(
        model=model, 
        device=device,
        dataset=test_dataset
    ) 
        
    results = generative_report(candidate_list, reference_list)
    
    results.to_csv(f"{run_path}/test_results.csv", index=False)
        
    logger.info("Saved generative report")
    
    with open(f"{run_path}/run.json", "w") as f:
        config = vars(args)
        config['train_loss'] = train_loss
        config['val_loss'] = val_loss
        config['run_id'] = run_id
        json.dump(config, f)

    os.remove(f"{ROOT}/{os.getenv('BLIP_CHECKPOINT')}") 