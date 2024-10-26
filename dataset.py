
from datasets import load_dataset
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
import logging
from datetime import datetime
from math import ceil
from random import randint as rand
from PIL import Image
import torch

now = datetime.now()
now = now.strftime("%Y-%m-%d")

random_seed = 42

np.random.seed(random_seed)
random.seed(random_seed)
torch.manual_seed(random_seed)

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
|---------------|
|GENERIC SECTION|
|---------------|
'''

'''
Function for transforming an id to a label
'''

def id2label(idx : int, classes: list) -> str:
    label = classes.keys()[classes.values().index(idx)]
    return label  

'''
Function for transforming a list of ids to a list of labels
'''

def id2label_list(idx_list : list) -> list:
    return list(map(id2label, idx_list))

'''
Function for transforming a single label to an id
'''

def label2id(label : str, classes: list) -> str:
    return classes.index(label)

'''
Function for converting a label list to an id list
'''

def label2id_list(label_list : list, classes : list) -> list:
    output = []

    for label in label_list:
        output.append(label2id(label, classes))

    return output

'''
|----------------------|
|IMAGE FEATURES SECTION|
|----------------------|
'''

'''
DataLoader for training the classifier on HyperKvasir and Kvasir Instrument
'''

class FeatureExtractor(Dataset):
    def __init__(self, path, label, code, bbox, train=False):  # Added parameter for the number of images per batch
        self.path = path
        self.label = label
        self.code = code
        self.bbox = bbox
        self.transform = None
        self.train = train 
         
        self.transform = transform()
        
        logging.info('Initialized Dataset')
    
    def __len__(self):
        return len(self.code)

    def __getitem__(self, idx):
        label = self.label[idx]
        path = self.path[idx]
        code = self.code[idx] 
        bbox = self.bbox[idx] 
        full_path = f"{path}/{code}.jpg"
        img = read_image(full_path)        

        if len(bbox) > 0:
            img = img[:, bbox[0]:bbox[1], bbox[2]:bbox[3]]

        transformed_image = self.transform(img.float())

        return transformed_image, label, path, code
    
'''
Utility function for handling the resulting .csv, in particular for the bounding
box column
'''

def format_bbox(x):
    if x == '[]':
        return []
    else:
        split = x[1:-1].split(',')
        bbox = [int(num) for num in split]
        return bbox
    
'''
Utility function that generates a json containing all the classes present in 
the feature extractor training data in the form of a dictionary as follows:

{
    class: numerical_index
}

'''

def generate_feature_extractor_classes_json(df : pd.DataFrame):
    class_names = df.label.unique()

    classes = {}

    for i in range(len(class_names)):
        classes[class_names[i]] = i

    with open(os.getenv('FEATURE_EXTRACTOR_CLASSES'), 'w') as f:
        json.dump(classes, f)

'''
Either loads the csv that contains all the images paths, labels and bounding boxes or
create said csv.
Optionally performs data augmentation on the images, applying random rotations and
horizontal flips.
'''

def prepare_feature_extractor_data(data_path, aug=False):
    if os.path.exists(data_path):
        df = pd.read_csv(data_path)

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

        if aug:
            logging.info('Augmenting data')

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
                            augmented_images.append([os.getenv('AUGMENTED_DATA'), l, p, bbox])

            df_aug = pd.DataFrame(augmented_images, columns=rows)

            logging.info(f"Created {len(augmented_images)} images")

            df = pd.concat((df, df_aug), axis=0)

        df = shuffle(df)

        df.to_csv(data_path, index=False)

        logging.info('Kvasir Df created')

        df['bbox'] = df['bbox'].apply(format_bbox)

        if not os.path.exists(os.getenv('FEATURE_EXTRACTOR_CLASSES')):
            generate_feature_extractor_classes_json(df)
    
    if not os.path.exists(os.getenv('FEATURE_EXTRACTOR_CLASSES')):
        generate_feature_extractor_classes_json(df)

    return df

'''
Utility function for splitting a Dataframe in two sets, 
by simply taking a user defined % of the dataset.
'''

def df_train_test_split(df : pd.DataFrame, test_size=0.2) -> pd.DataFrame:
    msk = np.random.rand(len(df)) < test_size
    train_set = df[~msk]
    test_set = df[msk]

    logging.info('Df split')
    
    return train_set, test_set

'''
Download the KVASIR-VQA dataset, splitted in the metadata.csv file and the imgs
'''

def load_kvasir_vqa() -> any:
    return load_dataset(os.getenv('KVASIR_VQA_DATASET'))

'''
Utility function for retrieving the original Kvasir VQA dataset using the HuggingFace Loader.
Hyper Kvasir and Kvasir Instrument classes are then remapped to match the classes present in Kvasir VQA.
'''

def retrieve_kvasir_vqa_dataset() -> None:
    dataset = load_kvasir_vqa()
    dataframe = dataset['raw'].select_columns(['source', 'question', 'answer', 'img_id']).to_pandas()
    
    if not os.path.exists(f"{os.getenv('KVASIR_VQA_METADATA')}/metadata.csv"):
        print('[LOG] Retrieving metadata.csv for Kvasir VQA')
        dataframe.to_csv(f"{os.getenv('KVASIR_VQA_METADATA')}/metadata.csv", index=False)
        
    if not os.path.exists(f"{os.getenv('KVASIR_VQA_DATA')}"):
        print('[LOG] Retrieving Kvasir VQA data')
        os.makedirs(f"{os.getenv('KVASIR_VQA_DATA')}", exist_ok=True)
        for i, row in dataframe.groupby('img_id').nth(0).iterrows(): # for images
            dataset['raw'][i]['image'].save(f"{os.getenv('KVASIR_VQA_DATA')}/{row['img_id']}.jpg")

'''
Utility function for loading the JSON file containing the mapping for the classes, in the form of:
{
    class: id
}
'''

def feature_extractor_classes():
    f = open(os.getenv('FEATURE_EXTRACTOR_CLASSES'))
    data = json.load(f)

    return data

'''
Utility function for retrieving a list of classes.
'''

def feature_extractor_class_names() -> list:
    classes = feature_extractor_classes()
    return list(classes.keys())

'''
Function for data image data augmentation. 
Num new images are created from the source image, applying a random
rotation and a random horizontal flip.
'''

def augment_image(src : str, code : str, num : int) -> list:

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

        new = f"{os.getenv('AUGMENTED_DATA')}/{new_code}.jpg"

        aug.save(new)
        data.append(new_code)

    return data

'''
Preprocessing function, applied before feeding images to the feature extractor.
'''

def transform() -> v2.Compose:

    transform = v2.Compose([
        v2.Resize((224,224)),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    return transform 

def main():
    retrieve_kvasir_vqa_dataset()
    
if __name__ == '__main__':
    main()