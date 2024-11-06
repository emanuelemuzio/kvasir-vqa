import sys
sys.path.append('src')

from dotenv import load_dotenv
import os
from transformers import AutoTokenizer, AutoModel
from dotenv import load_dotenv

load_dotenv()  
 
 
    
def get_tokenizer(model_name=os.getenv('LANGUAGE_MODEL')) -> AutoTokenizer:
    
    '''
    Tokenizer inizialization function.
    
    ----------
    Parameters
        model_name: str
            This will usually be empty, in favor of the .env defined language model
    ----------

    ------
    Return
        tokenizer: AutoTokenizer
            HuggingFace Tokenizer
    ------
    '''
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    return tokenizer



def get_language_model(model_name=os.getenv('LANGUAGE_MODEL')) -> AutoModel:
    
    '''
    Small utility function for recovering the model used for the word embeddings
    
    ----------
    Parameters
        model_name: str
            This will usually be empty, in favor of the .env defined language model
    ----------

    ------
    Return
        model: AutoModel
            HuggingFace Model
    ------
    '''
    
    model = AutoModel.from_pretrained(model_name)

    return model 



def encode_question(question : str, tokenizer=None, model=None, device='cpu'):
    
    '''
    Valutare rimozione e inserimento direttamente nelle funzioni di eval/inferenza
    '''
    
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