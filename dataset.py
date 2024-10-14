
from datasets import load_dataset
import os
import pandas as pd
from torchvision.io import read_image
from torch.utils.data import Dataset
from torchvision.transforms import v2
import torch
import numpy as np
import json
from dotenv import load_dotenv
from sklearn.utils import shuffle
import logging
from datetime import datetime
import cv2 as cv

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

class Kvasir(Dataset):
    def __init__(self, path, label, code, bbox, train=False):  # Added parameter for the number of images per batch
        self.path = path
        self.label = label
        self.code = code
        self.bbox = bbox
        # self.class_names = class_names
        self.transform = None
        self.train = train 
         
        # Define transformations
        if self.train:
            self.transform = v2.Compose([
                v2.RandomResizedCrop(224),
                v2.RandomHorizontalFlip(),
                v2.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05),
                v2.ToDtype(torch.float32, scale=True),
                v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
        else:
            self.transform = v2.Compose([
                v2.Resize((224, 224)),
                v2.ToDtype(torch.float32, scale=True),
                v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
        
        logging.info('Initialized Dataset')
    
    def __len__(self):
        return len(self.code)

    def __getitem__(self, idx):
        label = self.label[idx]
        path = self.path[idx]
        code = self.code[idx] 
        full_path = f"{path}/{code}.jpg"
        img = read_image(full_path)
        bbox = self.bbox[idx] 
        if len(bbox) > 0:
            img = img[:, bbox[0]:bbox[1], bbox[2]:bbox[3]]

        # Apply transformations
        transformed_image = self.transform(img)

        # cv_img = transformed_image.permute(1, 2, 0).numpy()
        # cv_img = cv.cvtColor(cv_img, cv.COLOR_BGR2RGB)

        # cv.imshow('test', cv_img)
        # cv.waitKey(0)
        # cv.destroyAllWindows()
        
        return transformed_image, label, path, code
    
def format_bbox(x):
    if x == '[]':
        return []
    else:
        split = x[1:-1].split(',')
        bbox = [int(num) for num in split]
        return bbox
    
# Prepare a Hyper-Kvasir df joint with Kvasir Instrument
         
def generate_kvasir_gradcam_classes_json(df : pd.DataFrame):
    class_names = df.label.unique()

    classes = {}

    for i in range(len(class_names)):
        classes[class_names[i]] = i

    with open(os.getenv('KVASIR_GRADCAM_CLASSES'), 'w') as f:
        json.dump(classes, f)

def prepare_data():
    if os.path.exists(os.getenv('KVASIR_GRADCAM_CSV')):
        df = pd.read_csv(os.getenv('KVASIR_GRADCAM_CSV'))

        df['bbox'] = df['bbox'].apply(format_bbox)

    else:

        rows = ['path','label','code','bbox']
        
        # Kvasir instrument load
        
        kvasir_inst_json_data = None
        kvasir_inst_data = []

        logging.info('Loading kvasir instruments')
        
        with open(os.getenv('KVASIR_INST_BBOX'),'r') as file:
            kvasir_inst_json_data = json.load(file)
            
        kvasir_inst_img_codes = kvasir_inst_json_data.keys()
        
        for code in kvasir_inst_img_codes:
            for bbox in kvasir_inst_json_data[code]['bbox']:
                path = os.getenv('KVASIR_INST_IMG')
                kvasir_inst_data.append(
                    [
                        path,
                        bbox['label'],
                        code,
                        f"[{bbox['ymin']},{bbox['ymax']},{bbox['xmin']},{bbox['xmax']}]"
                    ]
                ) 

        logging.info('Loaded kvasir instrument')

        logging.info('Loading hyper kvasir')

        # Hyper Kvasir Segmented images loading
        # The segmented ones are only related to the polyps class

        hyper_kvasir_segmented_json_data = None
        hyper_kvasir_data = []
        
        with open(os.getenv('HYPER_KVASIR_SEGMENTED_BBOX'),'r') as file:
            hyper_kvasir_segmented_json_data = json.load(file)
            
        hyper_kvasir_segmented_img_codes = hyper_kvasir_segmented_json_data.keys()
        
        for code in hyper_kvasir_segmented_img_codes:
            for bbox in hyper_kvasir_segmented_json_data[code]['bbox']:
                path = os.getenv('HYPER_KVASIR_SEGMENTED_IMG')
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
        
        with open(os.getenv('HYPER_KVASIR_LABELED_IMG_PATHS'),'r') as file:
            hyper_kvasir_labeled_img_paths = json.load(file)

        for path in hyper_kvasir_labeled_img_paths:
            label = path.split("/")
            label = label[-1]
            for code in os.listdir(f"{os.getenv('HYPER_KVASIR')}/{path}"):
                hyper_kvasir_data.append(
                    [
                        f"{os.getenv('HYPER_KVASIR')}/{path}",
                        label,
                        code[:-4],
                        "[]"
                    ]
                )

        logging.info('Loaded hyper kvasir')

        data = kvasir_inst_data + hyper_kvasir_data 

        df = pd.DataFrame(data, columns=rows) 
        df = shuffle(df)

        df.to_csv(os.getenv('KVASIR_GRADCAM_CSV'), index=False)

        logging.info('Kvasir Df created')

        df['bbox'] = df['bbox'].apply(format_bbox)

        if not os.path.exists(os.getenv('KVASIR_GRADCAM_CLASSES')):
            generate_kvasir_gradcam_classes_json(df)
    
    if not os.path.exists(os.getenv('KVASIR_GRADCAM_CLASSES')):
        generate_kvasir_gradcam_classes_json(df)

    return df

def df_train_test_split(df, test_size=0.2):
    msk = np.random.rand(len(df)) < test_size
    train_set = df[~msk]
    test_set = df[msk]

    logging.info('Df split')
    
    return train_set, test_set

'''
Download the KVASIR-VQA dataset, splitted in the metadata.csv file and the imgs
'''

def load() -> any:
    return load_dataset(os.getenv('KVASIR_VQA_DATASET'))

def retrieve_dataset() -> None:
    dataset = load()
    dataframe = dataset['raw'].select_columns(['source', 'question', 'answer', 'img_id']).to_pandas()
    
    if not os.path.exists(f"{os.getenv('KVASIR_VQA_METADATA')}/metadata.csv"):
        print('[LOG] Retrieving metadata.csv for Kvasir VQA')
        dataframe.to_csv(f"{os.getenv('KVASIR_VQA_METADATA')}/metadata.csv", index=False)
        
    if not os.path.exists(f"{os.getenv('KVASIR_VQA_DATA')}"):
        print('[LOG] Retrieving Kvasir VQA data')
        os.makedirs(f"{os.getenv('KVASIR_VQA_DATA')}", exist_ok=True)
        for i, row in dataframe.groupby('img_id').nth(0).iterrows(): # for images
            dataset['raw'][i]['image'].save(f"{os.getenv('KVASIR_VQA_DATA')}/{row['img_id']}.jpg")
            
def kvasir_gradcam_classes():
    f = open(os.getenv('KVASIR_GRADCAM_CLASSES'))
    data = json.load(f)

    return data

def kvasir_gradcam_class_names():
    classes = kvasir_gradcam_classes()
    return list(classes.keys())

def id2label(idx : int) -> str:
    classes = kvasir_gradcam_classes()
    label = classes.keys()[classes.values().index(idx)]
    return label

def id2label_list(idx_list : list) -> list:
    return list(map(id2label, idx_list))

def label2id(label : str) -> str:
    classes = kvasir_gradcam_classes()
    return classes[label]

def label2id_list(label_list : list) -> list:
    return list(map(label2id, label_list))

def main():
    retrieve_dataset()
    
if __name__ == '__main__':
    main()