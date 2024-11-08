import sys
sys.path.append('src')

import gc
from sklearn.preprocessing import LabelEncoder
import os
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
import torch
from common.util import ROOT, logger
from common.prompt_tuning import PromptTuning
import numpy as np
from dotenv import load_dotenv
from torcheval.metrics.functional import multiclass_accuracy
from common.earlystop import EarlyStopper
from classifier.data import Dataset_

load_dotenv() 



class Classifier(nn.Module):
    
    '''
    Core of the project, the classifier that uses the joint embeddings method.
    
    Builder parameters
    ------------------
        vocabulary_size: int
            N answers known to the model
        multimodal_fusion_dim: int
            Dimension of the multimodal fusion between the question encoding and
            the feature extraced from the image
        intermediate_dim: int
            output of the linear layer during used in the multimodal fusion
    ------------------
    
    Forward input
    ------
        concat_output: tensor
            Concat tensor that consists of the question and visual encodes
    ------
    
    Forward output
    ------
        logits: tensor
            Answers soft scores
    ------
    '''
    
    def __init__(
        self,
        vocabulary_size : int,
        question_embedding_dim : int,
        image_feature_dim : int,
        intermediate_dim=512):
        
        super(Classifier, self).__init__() 
        
        self.prepare_multimodal_v = nn.Sequential(
            nn.Linear(image_feature_dim, intermediate_dim),
            nn.ReLU(),
            nn.Dropout(0.5)
        )
        
        self.prepare_multimodal_q = nn.Sequential(
            nn.Linear(question_embedding_dim, intermediate_dim),
            nn.ReLU(),
            nn.Dropout(0.5)
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(intermediate_dim, vocabulary_size),
            nn.Sigmoid()
        )
        
    def forward(self, encoded_question, feature_vector): 
        
        v = self.prepare_multimodal_v(feature_vector)
        q = self.prepare_multimodal_q(encoded_question)
        
        h = torch.mul(v, q)
        
        logits = self.classifier(h)
        
        return logits 
    
    
    
def get_classifier(feature_extractor_name=None, vocabulary_size=0):
    
    '''
    Classifier initialization function
    
    Parameters
    ------------------
        feature_extractor_name: str
            Used for choosing the CNN base model for the feature extraction
        vocabulary_size: int
            Num. of answers that the classifier knows
    ------------------
    
    Return
    ------
        classifier: Classifier
            Initialized classifier model
    ------
    '''
    
    question_embedding_dim = int(os.getenv('EMBEDDING_DIM'))
    image_feature_dim = -1
    
    intermediate_dim = int(os.getenv('CLASSIFIER_INTERMEDIATE_DIM'))
    
    if feature_extractor_name.startswith('resnet'):
        image_feature_dim = int(os.getenv('RESNET_FEATURE_SIZE'))
    elif feature_extractor_name.startswith('vgg'):
        image_feature_dim = int(os.getenv('VGG_FEATURE_SIZE'))
    
    classifier = Classifier(
        vocabulary_size=vocabulary_size,
        question_embedding_dim=question_embedding_dim,
        image_feature_dim=image_feature_dim,
        intermediate_dim=intermediate_dim
    )
    
    return classifier



def evaluate(
            model : Classifier, 
            prompt_tuning : PromptTuning,
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
            best_acc_ckp = - np.inf,
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
        criterion: torch loss
            Torch Loss object, in this case Cross Entropy Loss
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

        epoch_train_loss, epoch_train_acc = train(
            model,
            prompt_tuning,
            train_dataloader, 
            criterion, 
            optimizer, 
            answer_encoder, 
            max_length, 
            tokenizer,
            question_encoder,
            feature_extractor,
            device, 
            scheduler
        )
        
        train_loss.append(epoch_train_loss)
        train_acc.append(epoch_train_acc)
        
        epoch_val_loss, epoch_val_acc = val(
            model, 
            prompt_tuning,
            val_dataloader, 
            criterion, 
            answer_encoder, 
            max_length, 
            question_encoder,
            tokenizer,
            feature_extractor, 
            device
        )
        
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
            }, f"{ROOT}/{os.getenv('CLASSIFIER_CHECKPOINT')}") 

            logger.info('Checkpoint reached')
            
        if early_stopper.early_stop(epoch_val_loss):
            logger.info('Early stop activating')
            break
            
    return train_loss, train_acc, val_loss, val_acc, best_acc, best_weights



def train(
    model : Classifier,
    prompt_tuning : PromptTuning, 
    train_dataset : Dataset_, 
    criterion : torch.nn, 
    optimizer : torch.optim, 
    answer_encoder : LabelEncoder, 
    max_length : int,
    tokenizer : AutoTokenizer,
    question_encoder : AutoModel,
    feature_extractor,
    device : str, 
    scheduler : list):
    
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

    for i, (img, question, answer) in enumerate(train_dataset):
        
        optimizer.zero_grad()

        prompt = prompt_tuning.generate(question=question)
        
        tuned_question = list(map(lambda x: x[0] + x[1], list(zip(question, prompt))))
        
        inputs = tokenizer(
                        tuned_question, 
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
        
        output = model(word_embeddings, image_feature)
        
        target = (torch.tensor(answer_encoder.transform(answer))).to(device)

        loss = criterion(output, target)
        loss.backward()
        optimizer.step() 
        
        acc = multiclass_accuracy(output, target) 
        train_acc += acc.item()
        train_loss += loss.item()
        
        gc.collect()
        
        if device == 'cuda':
            torch.cuda.empty_cache()

    for s in scheduler:
        s.step()

    train_loss = train_loss / len(train_dataset)
    train_acc = train_acc / len(train_dataset)

    return train_loss, train_acc



def val(
    model : Classifier, 
    prompt_tuning: PromptTuning,
    val_dataset : Dataset_, 
    criterion : torch.nn,  
    answer_encoder : LabelEncoder, 
    max_length : int,
    question_encoder : AutoModel,
    tokenizer : AutoTokenizer,
    feature_extractor,
    device : str):
    
    '''
    Validation step function
    
    ----------
    Parameters
        model: Classifier
            Classifier model
        val_dataset: torch Dataset
            Dataset object
        criterion: torch loss
            Torch Loss object, in this case Cross Entropy Loss
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
        for i, (img, question, answer) in enumerate(val_dataset):
            
            prompt = prompt_tuning.generate(question=question)
        
            tuned_question = list(map(lambda x: x[0] + x[1], list(zip(question, prompt))))
            
            inputs = tokenizer(
                        tuned_question, 
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
            
            output = model(word_embeddings, image_feature) 
            
            target = (torch.tensor(answer_encoder.transform(answer))).to(device)
            
            loss = criterion(output, target)
            
            acc = multiclass_accuracy(output, target)
            val_acc += acc.item()
            val_loss += loss.item()
            
            gc.collect()
        
            if device == 'cuda':
                torch.cuda.empty_cache()

        # Calculate validation loss
        val_loss /= len(val_dataset)

        # Calculate validation accuracy
        val_acc = val_acc / len(val_dataset) 

        return val_loss, val_acc 