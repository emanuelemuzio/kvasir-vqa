import sys
sys.path.append('src')

import torch
import os
import numpy as np
import pandas as pd
import argparse
import seaborn as sns
import json
from sklearn.preprocessing import LabelEncoder
from transformers import AutoTokenizer, AutoModel
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from common.util import ROOT, logger, get_run_info, generate_run_id, plot_run, check_run_type, get_new_tokens
from torcheval.metrics.functional import multiclass_accuracy
from common.earlystop import EarlyStopper
from custom.architecture import HadamardClassifier, ConcatClassifier, ConvVQA, BiggerConvVQA
from custom.data import Dataset_
from dotenv import load_dotenv
from question_encode.model import get_tokenizer, get_language_model
from feature_extractor.model import init
from torch.nn import CrossEntropyLoss
from sklearn.metrics import f1_score
from torch import optim
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR, LinearLR, ExponentialLR, StepLR
from sklearn.model_selection import train_test_split
import torch.nn.functional as F

load_dotenv()      
RANDOM_SEED = int(os.getenv('RANDOM_SEED'))
    
def get_classifier(feature_extractor_name=None, vocabulary_size=0, architecture=None, inference=False):
    
    question_embedding_dim = int(os.getenv('EMBEDDING_DIM'))
    image_feature_dim = -1
    
    intermediate_dim = int(os.getenv('CUSTOM_INTERMEDIATE_DIM'))
    
    if feature_extractor_name.startswith('resnet'):
        image_feature_dim = int(os.getenv('RESNET_FEATURE_SIZE'))
    elif feature_extractor_name.startswith('vgg'):
        image_feature_dim = int(os.getenv('VGG_FEATURE_SIZE'))
    elif feature_extractor_name.startswith('vit'):
        image_feature_dim = int(os.getenv('VIT_FEATURE_SIZE'))
        
    classifier = None
    
    if architecture == 'hadamard':
        classifier = HadamardClassifier(
        vocabulary_size=vocabulary_size,
        question_embedding_dim=question_embedding_dim,
        image_feature_dim=image_feature_dim,
        intermediate_dim=intermediate_dim,
        )
    elif architecture == 'concat':
        classifier = ConcatClassifier(
        vocabulary_size=vocabulary_size,
        question_embedding_dim=question_embedding_dim,
        image_feature_dim=image_feature_dim,
        intermediate_dim=intermediate_dim,
        )
    elif architecture == 'conv':
        classifier = ConvVQA(
            vocabulary_size=vocabulary_size
        )
    elif architecture == 'biggerconv':
        classifier = BiggerConvVQA(
            vocabulary_size=vocabulary_size
        )
        
    logger.info(f"Initialized {architecture} classifier")
        
    return classifier



def evaluate(
            model, 
            num_epochs : int, 
            batch_size : int, 
            optimizer : torch.optim, 
            scheduler : list, 
            device : str, 
            train_dataset : DataLoader, 
            val_dataset : DataLoader, 
            criterion : torch.nn, 
            early_stopper : EarlyStopper, 
            answer_encoder : LabelEncoder,
            max_length : int, 
            tokenizer : AutoTokenizer,
            feature_extractor,
            question_encoder : AutoModel,
            train_loss_ckp = [],
            train_acc_ckp = [],
            val_loss_ckp = [],
            val_acc_ckp = [],
            start_epoch = 1,
            run_id='test'):
    
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
            train_dataloader, 
            criterion, 
            optimizer, 
            answer_encoder, 
            max_length, 
            tokenizer,
            question_encoder,
            feature_extractor,
            device
        )
        
        train_loss.append(epoch_train_loss)
        train_acc.append(epoch_train_acc)
        
        epoch_val_loss, epoch_val_acc = val(
            model, 
            val_dataloader, 
            criterion, 
            answer_encoder, 
            max_length, 
            question_encoder,
            tokenizer,
            feature_extractor, 
            device,
            scheduler
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
            }, f"{ROOT}/{os.getenv('CUSTOM_RUNS')}/{run_id}/checkpoint.pt") 

            logger.info('Checkpoint reached')
            
        if early_stopper.early_stop(epoch_val_loss):
            logger.info('Early stop activating')
            break
            
    return train_loss, train_acc, val_loss, val_acc, best_weights



def train(
    model : ConcatClassifier|HadamardClassifier,
    train_dataset : Dataset_, 
    criterion : torch.nn, 
    optimizer : torch.optim, 
    answer_encoder : LabelEncoder, 
    max_length : int,
    tokenizer : AutoTokenizer,
    question_encoder : AutoModel,
    feature_extractor,
    device : str ):
    
    '''
    Train step function
    
    ----------
    Parameters
        model: Classifier
            Classifier model
        train_dataset: Dataset
            Dataset object
        criterion: torch loss
            Torch Loss object, in this case Cross Entropy Loss
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

    for i, (img, question, answer) in tqdm(enumerate(train_dataset)):
        
        optimizer.zero_grad() 
        
        inputs = tokenizer(
                        question, 
                        add_special_tokens=True, 
                        return_tensors='pt', 
                        padding='max_length', 
                        max_length=max_length, 
                        truncation=True).to(device)
        
        input_ids = inputs['input_ids'].to(device)
        token_type_ids = inputs['token_type_ids'].to(device)
        attention_mask = inputs['attention_mask'].to(device)

        word_embeddings = question_encoder(input_ids, attention_mask = attention_mask, token_type_ids = token_type_ids).last_hidden_state[:,0,:].squeeze().to(device)
        
        img = img.to(device)
        
        image_feature = feature_extractor(img).squeeze()
        
        logits = model(word_embeddings, image_feature)
        
        output = F.softmax(logits)
        
        target = (torch.tensor(answer_encoder.transform(answer))).to(device)

        loss = criterion(logits, target)
        loss.backward()
        optimizer.step() 
        
        acc = multiclass_accuracy(output, target) 
        train_acc += acc.item()
        train_loss += loss.item()
        
        if device == 'cuda':
            torch.cuda.empty_cache()

    train_loss = train_loss / len(train_dataset)
    train_acc = train_acc / len(train_dataset)

    return train_loss, train_acc



def val(
    model, 
    val_dataset : Dataset_, 
    criterion : torch.nn,  
    answer_encoder : LabelEncoder, 
    max_length : int,
    question_encoder : AutoModel,
    tokenizer : AutoTokenizer,
    feature_extractor,
    device : str,
    scheduler : list):
    
    # Evaluate the model on the validation set
    model.eval()
    val_loss = 0.0
    val_acc = 0.0
        
    with torch.no_grad():
        for i, (img, question, answer) in enumerate(val_dataset):
            
            inputs = tokenizer(
                        question, 
                        add_special_tokens=True, 
                        return_tensors='pt', 
                        padding='longest', 
                        max_length=max_length, 
                        truncation=True).to(device)
        
            input_ids = inputs['input_ids'].to(device)
            token_type_ids = inputs['token_type_ids'].to(device)
            attention_mask = inputs['attention_mask'].to(device)

            word_embeddings = question_encoder(input_ids, attention_mask = attention_mask, token_type_ids = token_type_ids).last_hidden_state[:,0,:].squeeze().to(device)
            
            img = img.to(device)
            
            image_feature = feature_extractor(img).squeeze()
            
            logits = model(word_embeddings, image_feature)
        
            output = F.softmax(logits)
            
            target = (torch.tensor(answer_encoder.transform(answer))).to(device)

            loss = criterion(logits, target)
            
            acc = multiclass_accuracy(output, target)
            val_acc += acc.item()
            val_loss += loss.item()
            
        
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
    dataset,
    tokenizer, 
    max_length,
    question_encoder,
    feature_extractor,
    answer_encoder
    ):
    
    model.eval()
    
    questions = []
    predictions = []
    targets = []
        
    with torch.no_grad():
        for i, (img, question, answer) in enumerate(dataset):
            
            inputs = tokenizer(
                        question, 
                        add_special_tokens=True, 
                        return_tensors='pt', 
                        padding='longest', 
                        max_length=max_length, 
                        truncation=True).to(device)
        
            input_ids = inputs['input_ids'].to(device)
            token_type_ids = inputs['token_type_ids'].to(device)
            attention_mask = inputs['attention_mask'].to(device)

            word_embeddings = question_encoder(input_ids, attention_mask = attention_mask, token_type_ids = token_type_ids).last_hidden_state[:,0,:].to(device)
            
            img = img.to(device).unsqueeze(0)
            
            image_feature = feature_extractor(img).squeeze(3).squeeze(-1)
            
            logits = model(word_embeddings, image_feature)
        
            output = F.softmax(logits)
            
            target = (torch.tensor(answer_encoder.transform([answer]))).item()

            predictions.append(torch.argmax(output).item())
            targets.append(target)  
            questions.append(question)          
        
            if device == 'cuda':
                torch.cuda.empty_cache()
                
    return questions, predictions, targets
    
    
def launch_experiment(args : argparse.Namespace, device: str) -> None:
    
    logger.info(f"Launching experiment with configuration: {args}")
    
    feature_extractor_run_path = f"{ROOT}/{os.getenv('FEATURE_EXTRACTOR_RUNS')}/{args.feature_extractor}/run.json"
    feature_extractor_weights_path = f"{ROOT}/{os.getenv('FEATURE_EXTRACTOR_RUNS')}/{args.feature_extractor}/model.pt"
    
    use_aug = args.use_aug == '1'
    
    feature_extractor_name = get_run_info(run_path=feature_extractor_run_path)['model'] 
    
    logger.info("Generating run id")
  
    run_id = generate_run_id()

    architecture = args.architecture 
    
    df = None
    
    kvasir_vqa_datapath = f"{ROOT}/{os.getenv('KVASIR_VQA_DATA')}"
    kvasir_vqa_datapath_aug = f"{ROOT}/{os.getenv('KVASIR_VQA_DATA_AUG')}"
    
    if use_aug:
        df = pd.read_csv(f"{ROOT}/{os.getenv('KVASIR_VQA_CSV_AUG')}")
    else:
        df = pd.read_csv(f"{ROOT}/{os.getenv('KVASIR_VQA_CSV_CLEAN')}") 
    
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
        test_size=0.3, 
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
        aug_path=kvasir_vqa_datapath_aug
    )
        
    test_dataset = Dataset_(
        source=X_test['source'].to_numpy(), 
        question=X_test['question'].to_numpy(), 
        answer=Y_test['answer'].to_numpy(), 
        img_id=X_test['img_id'].to_numpy(),
        base_path=kvasir_vqa_datapath,
        aug_path=kvasir_vqa_datapath_aug
    )
    
    val_dataset = Dataset_(
        source=X_val['source'].to_numpy(), 
        question=X_val['question'].to_numpy(), 
        answer=Y_val['answer'].to_numpy(), 
        img_id=X_val['img_id'].to_numpy(), 
        base_path=kvasir_vqa_datapath,
        aug_path=kvasir_vqa_datapath_aug
    )
    
    answers = list(set(df['answer']))
    
    answer_encoder = LabelEncoder().fit(answers)
    
    logger.info('Initializing classifier')
   
    model = get_classifier(feature_extractor_name=feature_extractor_name, vocabulary_size=len(answers), architecture=architecture, inference=False).to(device)
    
    tokenizer = get_tokenizer()
    question_encoder = get_language_model().to(device)
    feature_extractor = init(model_name=feature_extractor_name, weights_path=feature_extractor_weights_path).to(device)

    new_tokens = get_new_tokens(tokenizer)
    tokenizer.add_tokens(new_tokens)
    question_encoder.resize_token_embeddings(len(tokenizer))

    logger.info('Initialized tokenizer, question encoder and feature extractor')

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
    step_size = int(args.step_size) or None
    gamma = float(args.gamma) or None
    max_length = int(os.getenv('MAX_QUESTION_LENGTH'))
        
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
            if os.path.exists(f"{ROOT}/{os.getenv('CUSTOM_RUNS')}/{args.run_id}/checkpoint.pt"):
                os.remove(f"{ROOT}/{os.getenv('CUSTOM_RUNS')}/{args.run_id}/checkpoint.pt") 
        run_type = check_run_type(f"{ROOT}/{os.getenv('CUSTOM_RUNS')}/{args.run_id}")
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

    run_path = f"{ROOT}/{os.getenv('CUSTOM_RUNS')}/{run_id}"
    
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
                start_epoch = start_epoch,
                run_id=run_id)
                
            logger.info(f'Evaluation ended in {len(train_loss)} epochs')
                
            torch.save(best_weights, f"{run_path}/model.pt")
            
        case 2:
            min_epochs = 0
            checkpoint = torch.load(f"{ROOT}/{os.getenv('CUSTOM_RUNS')}/{run_id}/checkpoint.pt", weights_only=False)
            model.load_state_dict(checkpoint['model_state_dict'])
            start_epoch = checkpoint['num_epochs']
            train_loss_ckp = checkpoint['train_loss']
            train_acc_ckp = checkpoint['train_acc']
            val_loss_ckp = checkpoint['val_loss']
            val_acc_ckp = checkpoint['val_acc']
            run_id = checkpoint['run_id']
            run_path = f"{ROOT}/{os.getenv('CUSTOM_RUNS')}/{run_id}"
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
                start_epoch = start_epoch,
                run_id=run_id)
                
            logger.info(f'Evaluation ended in {len(train_loss)} epochs')
                
            torch.save(best_weights, f"{run_path}/model.pt")
            
        case 3: 
            val_acc = checkpoint['val_acc']
            val_loss = checkpoint['val_loss']
            train_acc = checkpoint['train_acc']
            train_loss = checkpoint['train_loss']
            best_weights = torch.load(f"{run_path}/model.pt", weights_only=True) 

    logger.info('Starting model evaluation') 
      
    logger.info(f'Starting test phase')
    
    model.load_state_dict(best_weights)
    
    question_list, y_pred, y_true = predict(
        model=model, 
        device=device,
        dataset=test_dataset,
        tokenizer=tokenizer, 
        max_length=max_length,
        question_encoder=question_encoder,
        feature_extractor=feature_extractor,
        answer_encoder=answer_encoder
    )
    
    f1_scoring = f1_score(y_true, y_pred, average=None).tolist()
    f1_macro_score = f1_score(y_true, y_pred, average='macro')
    
    columns = answers
    
    pd.DataFrame({'Class': columns, 'F1-Score': f1_scoring}).to_csv(f"{run_path}/test_results.csv", index=False)
    
    logger.info("Saved test report")
    
    with open(f"{run_path}/run.json", "w") as f:
        config = vars(args)
        config['train_loss'] = train_loss
        config['val_loss'] = val_loss
        config['train_acc'] = train_acc
        config['val_acc'] = val_acc
        config['run_id'] = run_id
        config['test_acc'] = f1_macro_score
        json.dump(config, f)

    os.remove(f"{run_path}/checkpoint.pt") 

    plot_run(base_path=f"{ROOT}/{os.getenv('CUSTOM_RUNS')}", run_id=run_id)
    
    logger.info("Run plotted and save to run.json file")