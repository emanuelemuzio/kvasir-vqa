import sys
sys.path.append('src')

import os
import pandas as pd
from torchvision.io import read_image
from torch.utils.data import Dataset
from torchvision.transforms import v2
import numpy as np
import json
import random
from dotenv import load_dotenv
from sklearn.utils import shuffle
from math import ceil
from random import randint as rand
from PIL import Image
import torch
from common.util import logger, ROOT, image_transform

RANDOM_SEED = int(os.getenv('RANDOM_SEED'))

np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

load_dotenv() 



class _Dataset(Dataset):
    
    '''
    PyTorch Dataset implementation for fine tuning the feature extractor block, used with
    data from Hyper Kvasir and Kvasir Instrument
    
    Builder parameters
    ------------------
        path: list
            List of paths to the image folder
        label: list
            List of ground truth label for the image
        code: list
            List of image codes for unique identification
        bbox: list/None
            List of bounding boxes, where present (only polyp and instrument images) 
    ------------------
    
    Return (with each evaluation loops)
    ------
        transformed_image: tensor
            224x224 normalized image, ready for inference or training
        label: str
            ground truth label for image
        path: str
            path to image folder
        code: str
            image unique identifier code
    ------
    '''
    
    def __init__(self, 
        path:list, 
        label:list, 
        code:list, 
        bbox:list
        ):
        
        self.path = path
        self.label = label
        self.code = code
        self.bbox = bbox
        self.transform = None
        self.transform = image_transform()
        
        logger.info('Initialized Feature Extractor Dataset')
    
    def __len__(self):
        return len(self.code)

    def __getitem__(self, idx:int):
        label = self.label[idx]
        path = self.path[idx]
        code = self.code[idx] 
        bbox = self.bbox[idx] 
        full_path = f"{ROOT}/{path}/{code}.jpg"
        img = read_image(full_path)        

        if len(bbox) > 0:
            img = img[:, bbox[0]:bbox[1], bbox[2]:bbox[3]]

        transformed_image = self.transform(img.float())

        return transformed_image, label, path, code
    
    
    
def format_bbox(x : str) -> list:
    
    '''
    Utility function for handling feature extractor .csv, in particular for the bounding box column
    
    Parameters
    ----------
        x: str
            may be either [] for not present bbox or [x, y, z, t]
    ----------
    
    Return
    ------
        bbox: list
            either empty list [] or [x, y, z, t]
    ------
    '''
    
    if x == '[]':
        return []
    else:
        split = x[1:-1].split(',')
        bbox = [int(num) for num in split]
        return bbox
     
     

def generate_classes_json(df : pd.DataFrame) -> None:
    
    '''
    Utility function that generates a json containing all the 
    classes present in the feature extractor training data 
    in the form of a dictionary.
    
    Parameters
    ----------
        df: DataFrame
            DataFrame that contains metadata for model evaluation.
    ---------- 
    '''
    
    class_names = df.label.unique()

    classes = {}

    for i in range(len(class_names)):
        classes[class_names[i]] = i

    with open(f"{ROOT}/{os.getenv('FEATURE_EXTRACTOR_CLASSES')}", 'w') as f:
        json.dump(classes, f)



def prepare_data(data_path:str, aug=False) -> pd.DataFrame:
    
    '''
    Either loads the csv that contains all the images paths, 
    labels and bounding boxes or create said csv.
    Optionally performs data augmentation on the images, 
    applying random rotations and horizontal flips.
    
    Parameters
    ----------
        data_path: str
            Csv path, which may or may not be present. If not present,
            it will be generated, accordingly to aug parameter too
        aug: bool, default:False
            True if you want to use augmented data/start data augmentation process, 
            else False

    ----------
    
    Return
    ------
        df: Dataframe
            DF that contains all the metadata for training the feature extractor
    ------
    '''
    
    df = None
    
    if os.path.exists(data_path):
        df = pd.read_csv(data_path)

        df['bbox'] = df['bbox'].apply(format_bbox)

    else:

        rows = ['path','label','code','bbox']
        
        # Kvasir instrument load
        
        kvasir_inst_json_data = None
        kvasir_inst_data = []

        logger.info('Loading kvasir instruments')
        
        with open(f"{ROOT}/{os.getenv('KVASIR_INST_BBOX')}",'r') as file:
            kvasir_inst_json_data = json.load(file)
            
        kvasir_inst_img_codes = kvasir_inst_json_data.keys()
        
        for code in kvasir_inst_img_codes:
            for bbox in kvasir_inst_json_data[code]['bbox']:
                path = f"{ROOT}/{os.getenv('KVASIR_INST_IMG')}"
                kvasir_inst_data.append(
                    [
                        path,
                        bbox['label'],
                        code,
                        f"[{bbox['ymin']},{bbox['ymax']},{bbox['xmin']},{bbox['xmax']}]"
                    ]
                ) 

        logger.info('Loaded kvasir instrument')

        logger.info('Loading hyper kvasir')

        # Hyper Kvasir Segmented images loading
        # The segmented ones are only related to the polyps class

        hyper_kvasir_segmented_json_data = None
        hyper_kvasir_data = []
        
        with open(f"{ROOT}/{os.getenv('HYPER_KVASIR_SEGMENTED_BBOX')}",'r') as file:
            hyper_kvasir_segmented_json_data = json.load(file)
            
        hyper_kvasir_segmented_img_codes = hyper_kvasir_segmented_json_data.keys()
        
        for code in hyper_kvasir_segmented_img_codes:
            for bbox in hyper_kvasir_segmented_json_data[code]['bbox']:
                path = f"{ROOT}/{os.getenv('HYPER_KVASIR_SEGMENTED_IMG')}"
                hyper_kvasir_data.append(
                    [
                        path,
                        bbox['label'],
                        code,
                        f"[{bbox['ymin']},{bbox['ymax']},{bbox['xmin']},{bbox['xmax']}]"
                    ]
                ) 

        # Labeled Kvasir Image question

        hyper_kvasir_labeled_img_paths = []
        
        with open(f"{ROOT}/{os.getenv('HYPER_KVASIR_LABELED_IMG_PATHS')}",'r') as file:
            hyper_kvasir_labeled_img_paths = json.load(file)

        for path in hyper_kvasir_labeled_img_paths:
            label = path.split("/")
            label = label[-1]
            for code in os.listdir(f"{ROOT}/{os.getenv('HYPER_KVASIR')}/{path}"):
                hyper_kvasir_data.append(
                    [
                        f"{ROOT}/{os.getenv('HYPER_KVASIR')}/{path}",
                        label,
                        code[:-4],
                        "[]"
                    ]
                )

        logger.info('Loaded hyper kvasir')

        data = kvasir_inst_data + hyper_kvasir_data 

        df = pd.DataFrame(data, columns=rows) 

        if aug:
            logger.info('Augmenting data')

            AUG_THRESHOLD = int(os.getenv('AUG_THRESHOLD'))
            labels = list(set(df["label"]))

            labels_count = {}

            for label in labels:
                labels_count[label] = len(df[df['label'] == label])

            augmented_images = []

            for l, n in labels_count.items():
                if n < AUG_THRESHOLD:
                    n_augs = ceil(AUG_THRESHOLD / n)
                    filt = df[df['label'] == l]
                        
                    for i in range(n):
                        row = filt.iloc[i]
                            
                        path = row['path']
                        label = row['label']
                        code = row['code']
                        bbox = row['bbox']

                        src = f"{path}/{code}.jpg"
                            
                        new_paths = augment_image(src, code, n_augs)

                        for p in new_paths:
                            augmented_images.append([f"{ROOT}/{os.getenv('AUGMENTED_DATA')}", l, p, bbox])

            df_aug = pd.DataFrame(augmented_images, columns=rows)

            logger.info(f"Created {len(augmented_images)} images")

            df = pd.concat((df, df_aug), axis=0)

        df = shuffle(df)

        df.to_csv(data_path, index=False)

        logger.info('Kvasir Df created')

        df['bbox'] = df['bbox'].apply(format_bbox)

        if not os.path.exists(f"{ROOT}/{os.getenv('FEATURE_EXTRACTOR_CLASSES')}"):
            generate_classes_json(df)
    
    if not os.path.exists(f"{ROOT}/{os.getenv('FEATURE_EXTRACTOR_CLASSES')}"):
        generate_classes_json(df)

    return df
 

 
def get_classes() -> dict:
 
    '''
    Utility function for loading the JSON file containing the mapping for the classes
     
    ------
    Return
        data: dict 
            JSON file dictionary as follows:
            {
                class: id
            }
    ------
    '''
    
    f = open(f"{ROOT}/{os.getenv('FEATURE_EXTRACTOR_CLASSES')}")
    data = json.load(f)

    return data 



def get_class_names() -> list:
    
    '''
    Utility function for class labels retrieve
    
    ------
    Return
        classes: list
            List of class names used for the feature extractor
    ------
    '''
    
    classes = get_classes()
    return list(classes.keys())



def augment_image(src : str, code : str, num : int) -> list:
    
    '''
    ------
    Parameters
        src: str
            Full path to image
        code: str
            Image unique identifier
        num: int
            Number of augmentations to perform per image
    ------
    
    ------
    Return
        data: list
            List that contains paths to all new images generated
    ------
    '''

    data = []

    random_rotations = []
    
    img = Image.open(src)

    for i in range(num):
        r_min = None
        r_max = None

        while True:
            r_min = rand(0, 180)
            r_max = rand(180, 270)

            if (r_min, r_max) not in random_rotations:
                random_rotations.append((r_min, r_max))
                break
        
        transform = v2.Compose([
                    v2.RandomHorizontalFlip(),
                    v2.RandomRotation((r_min, r_max))])
        
        aug = transform(img)

        new_code = f"{code}-aug-{i + 1}"

        new = f"{ROOT}/{os.getenv('AUGMENTED_DATA')}/{new_code}.jpg"

        aug.save(new)
        data.append(new_code)

    return data