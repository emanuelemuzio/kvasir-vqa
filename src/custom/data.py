from torchvision.io import read_image
from torch.utils.data import Dataset
from common.util import image_transform

class Dataset_(Dataset):
    
    '''
    Dataset class implementation for the Kvasir VQA Classifier.
        
    ----------
    Parameters
        source: list
            list of source paths to images
        question: list
            questions paired to images
        answer: list
            answer to questions
        img_id: list
            image identifier
        base_path: str
            base path to the correct folder
        prompt: str
            prompt that decorates the question
    ----------

    ------
    Return
        transformed_image: tensor
            Preprocessed image
        question: str
            question posed about the image, prompt tuned or not
        answer: str
            self explanatory at this point
    ------
    '''
    
    def __init__(self, source=[], question=[], answer=[], img_id=[],  base_path='', aug_path=''):  
        
        self.source = source
        self.question = question
        self.answer = answer
        self.img_id = img_id
        self.base_path = base_path
        self.aug_path = aug_path
        self.prompt = []
        self.transform = image_transform()
        self.use_prompt = len(self.prompt) > 0 
        
    def add_prompts(self, prompt=[]):
        self.prompt = prompt
    
    def __len__(self):
        return len(self.source)

    def __getitem__(self, idx):
        
        question = self.question[idx]
        answer = self.answer[idx]
        img_id = self.img_id[idx]
        use_prompt = self.use_prompt
        
        if use_prompt:
            question += self.prompt[idx]
        
        full_path = f"{self.aug_path}/{img_id}.jpg" if 'aug' in img_id else f"{self.base_path}/{img_id}.jpg" 
        img = read_image(full_path)         

        transformed_image = self.transform(img.float())

        return transformed_image, question, answer
    
class MultilabelDataset(Dataset):
    
    '''
    Dataset class implementation for the Kvasir VQA Classifier.
        
    ----------
    Parameters
        source: list
            list of source paths to images
        question: list
            questions paired to images
        answer: list
            answer to questions
        img_id: list
            image identifier
        base_path: str
            base path to the correct folder
        prompt: str
            prompt that decorates the question
    ----------

    ------
    Return
        transformed_image: tensor
            Preprocessed image
        question: str
            question posed about the image, prompt tuned or not
        answer: str
            self explanatory at this point
    ------
    '''
    
    def __init__(self, source=[], question=[], answer=[], img_id=[],  base_path=''):  
        
        self.source = source
        self.question = question
        self.answer = answer
        self.img_id = img_id
        self.base_path = base_path
        self.transform = image_transform()
    
    def __len__(self):
        return len(self.source)

    def __getitem__(self, idx):
        
        question = self.question[idx]
        answer = self.answer[idx]
        img_id = self.img_id[idx]
        
        full_path = f"{self.aug_path}/{img_id}.jpg" if 'aug' in img_id else f"{self.base_path}/{img_id}.jpg" 
        img = read_image(full_path)         

        transformed_image = self.transform(img.float())

        return transformed_image, question, answer