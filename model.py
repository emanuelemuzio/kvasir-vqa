from torchvision import models
from dotenv import load_dotenv
from torchvision import models
from datetime import datetime
import logging
from sklearn.preprocessing import LabelEncoder
import os
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
import torch
import numpy as np
from dotenv import load_dotenv
from torcheval.metrics.functional import multiclass_accuracy
from datetime import datetime
import logging
from dataset import label2id_list, feature_extractor_class_names, KvasirVQA
from callback import EarlyStopper

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

'''
|----------------------|
|IMAGE FEATURES SECTION|
|----------------------|
'''

'''
Function for retrieving the base model for either inference or training.
The model name is passed from the parameters.

Accepted model names are: resnet (50, 101 and 152) and vgg(16)

Inference =  0
Train only top layer = 1
Train all layers = 2
'''

def get_feature_extractor_model(model_name='resnet152', num_classes=0, freeze='2'):
    
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

'''
Evaluation function for the feature extractor, which includes both the train step and the validation
step for each epoch.
'''

def feature_extractor_evaluate(model, 
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
            class_names=[],
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
        epoch_train_loss, epoch_train_acc = feature_extractor_train(model, train_dataloader, criterion, optimizer, class_names, device, scheduler, scheduler_name)
        train_loss.append(epoch_train_loss)
        train_acc.append(epoch_train_acc)
        
        epoch_val_loss, epoch_val_acc = feature_extractor_val(model, val_dataloader, criterion, optimizer, class_names, device, scheduler, scheduler_name)
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
            }, os.getenv('FEATURE_EXTRACTOR_CHECKPOINT')) 

            logging.info('Checkpoint reached')
            
        if early_stopper.early_stop(epoch_val_loss):
            logging.info('Early stop activating')
            break
            
    return train_loss, train_acc, val_loss, val_acc, best_acc, best_weights

'''
Feature extractor train step function
'''

def feature_extractor_train(model, train_dataset, criterion, optimizer, class_names, device, scheduler, scheduler_name):
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

    if scheduler_name == 'cosine':
        scheduler.step()

    train_loss = train_loss / len(train_dataset)
    train_acc = train_acc / len(train_dataset)

    return train_loss, train_acc

'''
Feature extraction validation step
'''

def feature_extractor_val(model, val_dataset, criterion, optimizer, class_names, device, scheduler, scheduler_name):
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
        
        if scheduler_name == 'plateau':
            scheduler.step(val_loss)

        return val_loss, val_acc 
    
'''
Retrieve the model by name, load the weights and extract intermediate features 
from specific layers.

'''

def init_feature_extractor(model_name='resnet152', weights_path=os.getenv('FEATURE_EXTRACTOR_MODEL'), device='cpu'):
    class_names = feature_extractor_class_names()
    num_classes = len(class_names)

    model = get_feature_extractor_model(model_name=model_name, num_classes=num_classes)

    model.load_state_dict(torch.load(weights_path))
    
    model.eval()

    feature_extractor = None

    if model_name.startswith('resnet'):
        feature_extractor = torch.nn.Sequential(*list(model.children())[:-1])
    elif model_name.startswith('vgg'):
        model.classifier = model.classifier[:-1]
        feature_extractor = model

    feature_extractor.to(device)

    return feature_extractor

'''
|--------------------------|
|QUESTION EMBEDDING SECTION|
|--------------------------|
'''

'''
Tokenizer inizialization function
'''

def get_tokenizer(model_name=os.getenv('LANGUAGE_MODEL')):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    return tokenizer

'''
Small utility function for recovering the model used for the word embeddings
'''

def get_language_model(model_name=os.getenv('LANGUAGE_MODEL')):
    model = AutoModel.from_pretrained(model_name)

    return model

'''
Question encode function
'''

def encode_question(question : str, tokenizer=None, model=None, device='cpu'):
    model_name = os.getenv('LANGUAGE_MODEL')

    tokenizer = tokenizer or get_tokenizer(model_name=model_name)
    model = model or get_language_model(model_name=model_name).to(device)

    max_length = int(os.getenv('MAX_QUESTION_LENGTH'))
    inputs = tokenizer(question, 
                       add_special_tokens=True, 
                       return_tensors='pt', 
                       padding='max_length', 
                       max_length=max_length, 
                       truncation=True).to(device)

    input_ids = inputs['input_ids']
    attention_mask = inputs['attention_mask']
    token_type_ids = inputs['token_type_ids']

    outputs = model(input_ids, attention_mask = attention_mask, token_type_ids = token_type_ids)   
    word_embeddings = outputs.last_hidden_state
    
    word_embeddings = word_embeddings[:,0,:].squeeze()

    return word_embeddings

'''
|-----------|
|VQA SECTION|
|-----------|
'''

class VQAClassifier(nn.Module):
    def __init__(
        self, 
        vocabulary_size : int,
        multimodal_fusion_dim : int,
        intermediate_dim : int, 
        feature_extractor,
        tokenizer : AutoTokenizer, 
        question_encoder : AutoModel
    ):
        
        super(VQAClassifier, self).__init__()
            
        self.feature_extractor = feature_extractor 
        self.question_encoder = question_encoder
        self.tokenizer = tokenizer
        self.vocabulary_size = vocabulary_size
            
        self.multimodal_fusion = nn.Sequential(
            nn.Linear(multimodal_fusion_dim, intermediate_dim),
            nn.ReLU(),
            nn.Dropout(0.5)
        )
            
        self.classifier = nn.Sequential(
            nn.Linear(768, self.vocabulary_size),
        )

    def forward(self, question, preprocessed_image): 

        max_length = int(os.getenv('MAX_QUESTION_LENGTH'))
        inputs = self.tokenizer(question, 
                        add_special_tokens=True, 
                        return_tensors='pt', 
                        padding='max_length', 
                        max_length=max_length, 
                        truncation=True)

        input_ids = inputs['input_ids']
        token_type_ids = inputs['token_type_ids']
        attention_mask = inputs['attention_mask']

        question_encoding = self.question_encoder(input_ids, attention_mask = attention_mask, token_type_ids = token_type_ids).last_hidden_state
        
        question_encoding = question_encoding[:,0,:].squeeze()
        
        image_feature = self.feature_extractor(preprocessed_image).squeeze()
        
        fused_output = torch.cat([image_feature, question_encoding])
        
        logits = self.classifier(fused_output)
        
        return {
            'logits' : logits
        }
        
def get_vqa_classifier(language_model=os.getenv('LANGUAGE_MODEL'), feature_extractor_name=None, vocabulary_size=0):
    class_names = feature_extractor_class_names()
    
    multimodal_fusion_dim = 0
    
    intermediate_dim = int(os.getenv('CLASSIFIER_INTERMEDIATE_DIM'))
    
    if feature_extractor_name.startswith('resnet'):
        multimodal_fusion_dim = int(os.getenv('EMBEDDING_SIZE')) + int(os.getenv('RESNET_FEATURE_SIZE'))
    elif feature_extractor_name.startswith('vgg'):
        multimodal_fusion_dim = int(os.getenv('EMBEDDING_SIZE')) + int(os.getenv('VGG_FEATURE_SIZE'))
    
    tokenizer = get_tokenizer(language_model)
    question_encoder = get_language_model(language_model)
    feature_extractor = get_feature_extractor_model(model_name=feature_extractor_name, num_classes=len(class_names), freeze='0')
    
    classifier = VQAClassifier(
        vocabulary_size=vocabulary_size,
        multimodal_fusion_dim=multimodal_fusion_dim,
        intermediate_dim=intermediate_dim,
        feature_extractor=feature_extractor,
        tokenizer=tokenizer,
        question_encoder=question_encoder
    )
    
    return classifier

'''
Evaluation function for the feature extractor, which includes both the train step and the validation
step for each epoch.
'''

def classifier_evaluate(model : VQAClassifier, 
            num_epochs : int, 
            batch_size : int, 
            optimizer : torch.optim, 
            scheduler : torch.optim.lr_scheduler,
            scheduler_name : str,
            device : str, 
            train_dataset : DataLoader, 
            val_dataset : DataLoader, 
            criterion : torch.nn, 
            early_stopper : EarlyStopper, 
            answer_encoder : LabelEncoder,
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
        epoch_train_loss, epoch_train_acc = classifier_train(model, train_dataloader, criterion, optimizer, answer_encoder, device, scheduler, scheduler_name)
        train_loss.append(epoch_train_loss)
        train_acc.append(epoch_train_acc)
        
        epoch_val_loss, epoch_val_acc = classifier_val(model, val_dataloader, criterion, optimizer, answer_encoder, device, scheduler, scheduler_name)
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
            }, os.getenv('CLASSIFIER_CHECKPOINT')) 

            logging.info('Checkpoint reached')
            
        if early_stopper.early_stop(epoch_val_loss):
            logging.info('Early stop activating')
            break
            
    return train_loss, train_acc, val_loss, val_acc, best_acc, best_weights

'''
Classifier train step function
'''

def classifier_train(
    model : VQAClassifier, 
    train_dataset : KvasirVQA, 
    criterion : torch.nn, 
    optimizer : torch.optim, 
    answer_encoder : LabelEncoder, 
    device : str, 
    scheduler : torch.optim.lr_scheduler, 
    scheduler_name : str
    ):
    
    model.train()

    train_loss = 0.0
    train_acc = 0.0

    for i, (img, question, answer) in enumerate(train_dataset):

        img = img.to(device)
        target = (torch.tensor(answer_encoder.transform(answer))).to(device)

        optimizer.zero_grad()

        output = model(question, img)

        loss = criterion(output, target)

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

'''
Classifier validation step
'''

def classifier_val(model, val_dataset, criterion, optimizer, class_names, device, scheduler, scheduler_name):
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
        
        if scheduler_name == 'plateau':
            scheduler.step(val_loss)

        return val_loss, val_acc 