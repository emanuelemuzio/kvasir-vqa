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
# import cv2 as cv
import warnings
import random
import matplotlib.pyplot as plt
import json
from dotenv import load_dotenv

load_dotenv()

# class CustomKvasir(Dataset):
#     def __init__(self, img, labels, class_names, num_images=50, train=False):  # Added parameter for the number of images per batch
#         self.img = img
#         self.label = labels
#         self.class_names = class_names
#         self.num_images = num_images  # Store the number of images per batch
#         self.transform = None
#         self.train = train
        
#         # Define class names
#         # self.class_names = {0: 'apple', 1: 'banana', 2: 'grape', 3: 'orange', 4: 'pineapple', 5: 'watermelon'}
        
#         # Define transformations
#         if self.train:
#             self.transform = transforms.Compose([
#                 transforms.RandomResizedCrop(240),
#                 transforms.RandomHorizontalFlip(),
#                 transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
#                 transforms.ToTensor(),
#                 transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
#             ])
#         else:
#             self.transform = transforms.Compose([
#                 transforms.Resize(224),
#                 transforms.ToTensor(),
#                 transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
#             ])
    
#     def __len__(self):
#         return len(self.img)

#     def __getitem__(self, idx):
#         img = self.img[idx]
#         label = self.label[idx]
        
#         # Apply transformations
#         transformed_image = self.transform(Image.fromarray(img))
        
#         return {
#             'img' : transformed_image,
#             'label' : label
#         }
      
def prepare_data():
    rows = ['code','label','xmin','xmax','ymin','ymax']
    
    
    # Kvasir instrument load
    
    kvasir_inst_json_data = None
    kvasir_inst_data = []
    
    with open(os.getenv('KVASIR_INST_BBOX'),'r') as file:
        kvasir_inst_json_data = json.load(file)
        
    kvasir_inst_img_codes = kvasir_inst_json_data.keys()
    
    for code in kvasir_inst_img_codes:
        for bbox in kvasir_inst_json_data[code]['bbox']:
            kvasir_inst_data.append(
                [
                    code,
                    bbox['label'],
                    bbox['xmin'],
                    bbox['xmax'],
                    bbox['ymin'],
                    bbox['ymax'],
                ]
            )
        
    df = pd.DataFrame(kvasir_inst_data, columns=rows)
    
    print(1)
    
prepare_data()    


# exit(0)        
    
# train_dataset = MultiLabelDataset(num_images=200)
# val_dataset = MultiLabelDataset(sub_root='valid', num_images=50, train=False)

# for i, data in enumerate(train_dataset):
#     images, labels = data[0], data[1]
#     class_index = torch.argmax(labels[0])  # Get index of the first non-zero label
#     class_name = train_dataset.class_names[class_index.item()]  # Convert index to class name
#     plt.title(class_name)
#     plt.imshow(np.transpose(images.numpy(), (1, 2, 0)))  # Transpose the image to (240, 240, 3)
#     plt.show()
    
#     # Obtain the path of the plotted image
#     image_path = train_dataset.get_image_path(i)
#     # Read the original image using OpenCV
#     original_image = cv2.imread(image_path)
#     original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)

#     # Plot the original image
#     plt.title("Original Image")
#     plt.imshow(original_image)
#     plt.axis('off')
    
#     # Obtain the labels of the plotted image
#     image_labels = train_dataset.get_labels(i)
#     print("Labels:", image_labels)
#     break

# for i, data in enumerate(val_dataset):
#     images, labels = data[0], data[1]
#     class_index = torch.argmax(labels[0])  # Get index of the first non-zero label
#     class_name = val_dataset.class_names[class_index.item()]  # Convert index to class name
#     plt.title(class_name)
#     plt.imshow(np.transpose(images.numpy(), (1, 2, 0)))  # Transpose the image to (240, 240, 3)
#     plt.show()
#     # Obtain the path of the plotted image
#     image_path = val_dataset.get_image_path(i)
#     # Read the original image using OpenCV
#     original_image = cv2.imread(image_path)
#     original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)

#     # Plot the original image
#     plt.title("Original Image")
#     plt.imshow(original_image)
#     plt.axis('off')
    
#     # Obtain the labels of the plotted image
#     image_labels = train_dataset.get_labels(i)
#     print("Labels:", image_labels)
#     break

# from torch import nn, optim
# from torchvision import models

# # Initialize pre-trained model
# pretrained_model = models.resnet50(pretrained=True)

# # Iterate through the named modules and print each one
# for name, layer in pretrained_model.named_modules():
#     display(name, layer)
    
# # Get the last layer of the model
# last_layer_name, last_layer = list(pretrained_model.named_children())[-1]
# # Print information about the last layer
# print("Last Layer Name:", last_layer_name)
# print("Last Layer:", last_layer)

# num_classes = 6

# pretrained_model.fc = nn.Linear(pretrained_model.fc.in_features, num_classes)

# # Freeze the parameters of the pre-trained layers
# for param in pretrained_model.parameters():
#     param.requires_grad = False

# # Unfreeze the parameters of the last few layers for fine-tuning
# for param in pretrained_model.layer4.parameters():
#     param.requires_grad = True

# last_layer_name, last_layer = list(pretrained_model.layer4())
# print("New last layer:", last_layer)

# last_layer_name, last_layer = list(pretrained_model.named_children())[-1]
# print("New last layer:", last_layer)

# from torch.nn import BCEWithLogitsLoss
# import torch.optim as optim

# # Define loss function (Binary Cross Entropy Loss in this case, for multi-label classification)
# criterion = BCEWithLogitsLoss()

# # Define optimizer
# optimizer = optim.SGD(pretrained_model.parameters(), lr=0.001, momentum=0.9)

# # Training loop
# num_epochs = 50
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# pretrained_model.to(device)

# train_losses = []  # To store the losses for plotting
# best_val_loss = float('inf')  # Initialize with a very large value

# # Train the model
# for epoch in range(num_epochs):
    
#     # Train the model on the training set
#     pretrained_model.train()
    
#     # Initialize the training loss accumulator to zero
#     training_loss = 0.0
    
#     for i, (image, labels) in enumerate(train_dataset):
#         # Prepare data and send it to the proper device
#         image = image.unsqueeze(0).to(device)
#         labels = labels.float().to(device)

#         # Clear the gradients of all optimized parameters
#         optimizer.zero_grad()

#         # Forward pass: obtain model predictions for the input data
#         outputs = pretrained_model(image)

#         # Compute the loss between the model predictions and the true labels
#         loss = criterion(outputs, labels)

#         # Backward pass: compute gradients of the loss with respect to model parameters
#         loss.backward()

#         # Update model parameters using the computed gradients and the optimizer
#         optimizer.step()

#         # Update the training loss
#         training_loss += loss.item()

#     # Calculate average training loss
#     train_loss = training_loss / len(train_dataset)
#     train_losses.append(train_loss)

#     # Evaluate the model on the validation set
#     pretrained_model.eval()
#     val_loss = 0.0
#     correct_preds = 0
#     total_samples = 0
#     with torch.no_grad():
#         for image, labels in val_dataset:
#             # Prepare data and send it to the proper device
#             image = image.unsqueeze(0).to(device)
#             labels = labels.float().to(device)

#             # Forward pass: obtain model predictions for the input data
#             outputs = pretrained_model(image)

#             # Compute the loss between the model predictions and the true labels
#             loss = criterion(outputs, labels)

#             # Update the validation loss
#             val_loss += loss.item()

#             # Round up and down to either 1 or 0
#             predicted = torch.round(outputs)
#             total_samples += labels.size(0)
#             # Calculate how many images were correctly classified
#             correct_preds += torch.sum(torch.all(torch.eq(predicted, labels), dim=1)).item()

#     # Calculate validation loss
#     val_loss /= len(val_dataset)

#     # Calculate validation accuracy
#     val_acc = correct_preds / total_samples * 100

#     # Print validation loss and accuracy
#     print(f"Epoch [{epoch + 1}/{num_epochs}] Train Loss: {train_loss:.4f}  Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_acc:.2f}%")

#     # Save the model if it performs better on validation set
#     if val_loss < best_val_loss:
#         best_val_loss = val_loss
#         torch.save(pretrained_model.state_dict(), f'model/train/best_model_epoch_{epoch + 1}.pth')

# print('Finished Training')

# # Plotting the evolution of loss
# plt.plot(train_losses, label='Training Loss')
# plt.xlabel('Epoch')
# plt.ylabel('Loss')
# plt.title('Evolution of Training Loss')
# plt.legend()
# plt.show()