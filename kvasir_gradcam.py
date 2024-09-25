# Fine tuning a ResNet50 on hyper kvasir and kvasir inst to match the data used for Kvasir VQA

import os
import pandas as pd
from torchvision.io import read_image
from PIL import Image, ImageDraw
from torch.utils.data import Dataset, WeightedRandomSampler, DataLoader
from torchvision import transforms, models
from tqdm.auto import tqdm
import torch
import numpy as np
import cv2
import warnings
import random
import matplotlib.pyplot as plt

class KvasirJointDataset(Dataset):
    def __init__(self, 
                 img_root='Fruits-detection', 
                 sub_root='train', 
                 img_dir='images',
                 img_labels='labels',
                 train=True,
                 num_images=50):  # Added parameter for the number of images per batch
        self.img_root = img_root
        self.sub_root = sub_root
        self.img_dir = img_dir
        self.img_labels = img_labels
        self.train = train
        self.num_images = num_images  # Store the number of images per batch
        
        # Define class names
        self.class_names = {0: 'apple', 1: 'banana', 2: 'grape', 3: 'orange', 4: 'pineapple', 5: 'watermelon'}

        # Initialize an array of zeros for multi-label encoding
        self.multi_label_labels = []

        # Get labels
        dfs = []
        self.labels_path = os.path.join(self.img_root, self.sub_root, self.img_labels)
        for file in os.listdir(self.labels_path):
            if file.endswith(".txt"):
                file_path = os.path.join(self.labels_path, file)
                df = pd.read_csv(file_path, delimiter=" ", names=["label", "coordinate_1", "coordinate_2", "coordinate_3", "coordinate_4"])  # Adjust column names as needed
                df["filename"] = os.path.splitext(file)[0] + '.jpg'
                df["class_name"] = df["label"].map(self.class_names)
                dfs.append(df)

        self.final_df = pd.concat(dfs, ignore_index=True)
        
        # Get class-wise indices to randomly select images
        self.class_indices = {class_name: self.final_df[self.final_df['class_name'] == class_name].index.tolist() for class_name in self.class_names.values()}
                
        # Initialize an empty list to store multi-label encoded labels
        multi_label_labels_dict = {}

        # Randomly select images for each class
        selected_indices = []
        for class_name, indices in self.class_indices.items():
            selected_indices.extend(random.sample(indices, min(len(indices), self.num_images)))

        # Iterate over the selected indices
        for idx in selected_indices:
            filename = self.final_df.loc[idx, 'filename']
            labels = set(self.final_df[self.final_df['filename'] == filename]['label'].tolist()) # If there are multiple fruits in an image
            
            # Initialize an array of zeros for multi-label encoding
            multi_label_labels = [0] * len(self.class_names)
            
            # Set the corresponding indices to 1 for each label
            for label in labels:
                multi_label_labels[label] = 1

            # Append the multi-label encoded labels to the dictionary
            multi_label_labels_dict[filename] = [multi_label_labels]
        
        # Convert the dictionary of lists to a list of arrays
        multi_label_labels_list = [torch.tensor(labels) for labels in multi_label_labels_dict.values()]

        # Stack the padded tensors to create a single tensor
        self.multi_label_labels_tensor = torch.stack(multi_label_labels_list)

        # Get images
        self.imgs_path = os.path.join(self.img_root, self.sub_root, self.img_dir)
        image_names = self.final_df.iloc[selected_indices]['filename'].tolist()
        self.images = []
        self.images_path = []
        for img_name in tqdm(image_names):
            image_path = os.path.join(self.imgs_path, img_name)
            image = cv2.imread(image_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            self.images_path.append(image_path)
            self.images.append(image)
        
        # Define transformations
        if self.train:
            self.transform = transforms.Compose([
                transforms.RandomResizedCrop(240),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
    
    def get_image_path(self, idx):
        img_path = self.images_path[idx] 
        return img_path
    
    def get_labels(self, idx):
        return self.multi_label_labels_tensor[idx]
    
    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.multi_label_labels_tensor[idx]
        
        # Apply transformations
        transformed_image = self.transform(Image.fromarray(image))
        
        return transformed_image, label