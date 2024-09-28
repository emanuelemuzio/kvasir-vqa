# Fine tuning a ResNet50 on hyper kvasir and kvasir inst to match the data used for Kvasir VQA

import os
import pandas as pd
from torchvision.io import read_image
from torch.utils.data import Dataset, DataLoader
from torchvision import models
from torchvision.transforms import v2
from torchvision.models import ResNet50_Weights
from tqdm.auto import tqdm
import torch
import numpy as np
import json
from dotenv import load_dotenv
from torch import nn, optim
from torchvision import models
from torch.nn import CrossEntropyLoss
import torch.optim as optim
from sklearn.preprocessing import LabelEncoder
from torcheval.metrics.functional import multiclass_accuracy

load_dotenv()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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
                v2.RandomResizedCrop(240),
                v2.RandomHorizontalFlip(),
                v2.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
                v2.ToDtype(torch.float32, scale=True),
                v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
        else:
            self.transform = v2.Compose([
                v2.Resize((224, 224)),
                v2.ToDtype(torch.float32, scale=True),
                v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
    
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
        
        return transformed_image, label, path, code 
    
class EarlyStopper:
    def __init__(self, patience=1, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = float('inf')

    def early_stop(self, validation_loss):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False
      
def prepare_data():
    rows = ['path','label','code','bbox']
    
    # Kvasir instrument load
    
    kvasir_inst_json_data = None
    kvasir_inst_data = []
    
    with open(os.getenv('KVASIR_INST_BBOX'),'r') as file:
        kvasir_inst_json_data = json.load(file)
        
    kvasir_inst_img_codes = kvasir_inst_json_data.keys()
    
    for code in kvasir_inst_img_codes:
        for bbox in kvasir_inst_json_data[code]['bbox']:
            # im = cv.imread(f"{os.getenv('KVASIR_INST_IMG')}/{code}.jpg")
            # im = im[bbox['ymin']:bbox['ymax'], bbox['xmin']:bbox['xmax']]
            # im = cv.cvtColor(im, cv.COLOR_BGR2RGB)
            path = os.getenv('KVASIR_INST_IMG')
            kvasir_inst_data.append(
                [
                    path,
                    bbox['label'],
                    code,
                    [bbox['ymin'], bbox['ymax'], bbox['xmin'], bbox['xmax']]
                ]
            ) 

    # Hyper Kvasir Segmented images loading
    # The segmented ones are only related to the polyps class

    hyper_kvasir_segmented_json_data = None
    hyper_kvasir_data = []
    
    with open(os.getenv('HYPER_KVASIR_SEGMENTED_BBOX'),'r') as file:
        hyper_kvasir_segmented_json_data = json.load(file)
        
    hyper_kvasir_segmented_img_codes = hyper_kvasir_segmented_json_data.keys()
    
    for code in hyper_kvasir_segmented_img_codes:
        for bbox in hyper_kvasir_segmented_json_data[code]['bbox']:
            # im = cv.imread(f"{os.getenv('KVASIR_INST_IMG')}/{code}.jpg")
            # im = im[bbox['ymin']:bbox['ymax'], bbox['xmin']:bbox['xmax']]
            # im = cv.cvtColor(im, cv.COLOR_BGR2RGB)
            path = os.getenv('HYPER_KVASIR_SEGMENTED_IMG')
            hyper_kvasir_data.append(
                [
                    path,
                    bbox['label'],
                    code,
                    [bbox['ymin'], bbox['ymax'], bbox['xmin'], bbox['xmax']]
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
                    []
                ]
            )

    data = kvasir_inst_data + hyper_kvasir_data 

    df = pd.DataFrame(data, columns=rows) 
    
    return df

def prepare_pretrained_model(num_classes):
    # # Initialize pre-trained model
    pretrained_model = models.resnet50(weights=ResNet50_Weights.DEFAULT)
    pretrained_model.fc = nn.Linear(pretrained_model.fc.in_features, num_classes)

    # # Freeze the parameters of the pre-trained layers
    for param in pretrained_model.parameters():
        param.requires_grad = False

    # # Unfreeze the parameters of the last few layers for fine-tuning
    for param in pretrained_model.layer4.parameters():
        param.requires_grad = True

    return pretrained_model

def df_train_test_split(df, test_size=0.2):
    msk = np.random.rand(len(df)) < test_size
    train_set = df[~msk]
    test_set = df[msk]
    return train_set, test_set

def evaluate(model, 
            num_epochs, 
            batch_size, 
            optimizer, 
            device, 
            train_dataset, 
            val_dataset, 
            criterion, 
            early_stopper, 
            encoder, 
            train_loss_ckp = [],
            train_acc_ckp = [],
            val_loss_ckp = [],
            val_acc_ckp = [],
            best_acc_ckp = - np.inf,
            start_epoch = 1):
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
    
    train_loss = []
    val_loss = []

    train_acc = []
    val_acc = []
    best_acc = - np.inf
    best_weights = None

    if train_loss_ckp is not None:
        train_loss += train_loss_ckp
        train_acc += train_acc_ckp
        val_loss += val_loss_ckp
        val_acc += val_acc_ckp
        best_acc = best_acc_ckp

    # Ciclo di addestramento

    for epoch in tqdm(range(start_epoch, num_epochs + 1)):
        torch.cuda.empty_cache()
        epoch_train_loss, epoch_train_acc = train(model, train_dataloader, criterion, optimizer, device, encoder)
        train_loss.append(epoch_train_loss)
        train_acc.append(epoch_train_acc)
        
        epoch_val_loss, epoch_val_acc = val(model, val_dataloader, criterion, device, encoder)
        val_loss.append(epoch_val_loss)
        val_acc.append(epoch_val_acc)
        
        print(f"\nEpoch [{epoch}/{num_epochs}] Train Loss: {epoch_train_loss:.4f}  Validation Loss: {epoch_val_loss:.4f}, Validation Accuracy: {epoch_val_acc*100:.2f}%")
        
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
                'best_acc' : best_acc
            }, os.getenv('KVASIR_GRADCAM_CHECKPOINT'))

        # train_loss_ckp = checkpoint['train_loss']
        # train_acc_ckp = checkpoint['train_acc']
        # val_loss_ckp = checkpoint['val_loss']
        # val_acc_ckp = checkpoint['val_acc']
        # best_acc_ckp = checkpoint['best_acc']
            
        if early_stopper.early_stop(epoch_val_loss):             
            break
            
    return train_loss, train_acc, val_loss, val_acc, best_acc, best_weights
        
def train(model, train_dataset, criterion, optimizer, device, encoder):
    model.train()

    train_loss = 0.0
    train_acc = 0.0

    for i, (img, label, __, _) in enumerate(train_dataset):

        img = img.unsqueeze(0).to(device)[0]
        label = torch.tensor(encoder.transform(label)).to(device)

        optimizer.zero_grad()

        output = model(img)

        loss = criterion(output, label)

        loss.backward()

        optimizer.step()

        acc = multiclass_accuracy(output, label) 
        train_acc += acc.item()
        train_loss += loss.item()

    train_loss = train_loss / len(train_dataset)
    train_acc = train_acc / len(train_dataset)

    return train_loss, train_acc

def val(model, val_dataset, criterion, device, encoder):
    # Evaluate the model on the validation set
    model.eval()
    val_loss = 0.0
    val_acc = 0.0
        
    with torch.no_grad():
        for i, (img, label, _, _) in enumerate(val_dataset):
            img = img.unsqueeze(0).to(device)[0]
            label = torch.tensor(encoder.transform(label)).to(device)

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
        
        return val_loss, val_acc 
 
if __name__ == '__main__':
    dataset = prepare_data()    
    class_names = dataset.label.unique()
    enc = LabelEncoder()
    enc.fit(class_names)
    train_set, val_set = df_train_test_split(dataset, 0.2)
    train_dataset = Kvasir(train_set['path'].to_numpy(), train_set['label'].to_numpy(), train_set['code'].to_numpy(), train_set['bbox'].to_numpy(), train=True)
    val_dataset = Kvasir(val_set['path'].to_numpy(), val_set['label'].to_numpy(), val_set['code'].to_numpy(), val_set['bbox'].to_numpy(), train=False)

    num_classes = len(class_names)

    pretrained_model = prepare_pretrained_model(num_classes)
    
    # Define loss function (Binary Cross Entropy Loss in this case, for multi-label classification)
    criterion = CrossEntropyLoss()

    lr = 0.001
    momentum = 0.9

    # # Define optimizer
    optimizer = optim.SGD(pretrained_model.parameters(), lr=lr, momentum=momentum)
    
    early_stopper = EarlyStopper(patience=3, min_delta=0.5)

    # # Training loop
    num_epochs = 50
    batch_size = 18
    
    pretrained_model.to(device)

    train_loss_ckp = None
    train_acc_ckp = None
    val_loss_ckp = None
    val_acc_ckp = None
    best_acc_ckp = None 
    start_epoch = 1

    if os.path.exists(os.getenv('KVASIR_GRADCAM_CHECKPOINT')):
        checkpoint = torch.load(os.getenv('KVASIR_GRADCAM_CHECKPOINT'), weights_only=True)
        pretrained_model.state_dict(checkpoint['model_state_dict'])
        start_epoch = checkpoint['num_epochs']
        train_loss_ckp = checkpoint['train_loss']
        train_acc_ckp = checkpoint['train_acc']
        val_loss_ckp = checkpoint['val_loss']
        val_acc_ckp = checkpoint['val_acc']
        best_acc_ckp = checkpoint['best_acc']

    train_loss, train_acc, val_loss, val_acc, best_acc, best_weights = evaluate(
        model=pretrained_model, 
        num_epochs=num_epochs, 
        batch_size=batch_size,
        optimizer=optimizer, 
        device=device, 
        train_dataset=train_dataset, 
        val_dataset=val_dataset, 
        criterion=criterion, 
        early_stopper=early_stopper, 
        encoder=enc,
        train_loss_ckp = train_loss_ckp,
        train_acc_ckp = train_acc_ckp,
        val_loss_ckp = val_loss_ckp,
        val_acc_ckp = val_acc_ckp,
        best_acc_ckp = best_acc_ckp,
        start_epoch = start_epoch)
    
    torch.save(best_weights, os.getenv('KVASIR_GRADCAM_MODEL'))

    # Plotting the evolution of loss
    # plt.plot(train_losses, label='Training Loss')
    # plt.xlabel('Epoch')
    # plt.ylabel('Loss')
    # plt.title('Evolution of Training Loss')
    # plt.legend()
    # plt.show()