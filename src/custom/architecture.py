import sys
sys.path.append('src')

import torch
import torch.nn as nn
import torch.nn.functional as F
from common.prompting import PromptTuning

class HadamardClassifier(nn.Module):
    
    '''
    Classifier architecture which operates a Hadamard product between the
    feature extractor and the word embedding model outputs.
    
    Builder parameters
    ------------------
        vocabulary_size: int
            N answers known to the model
        multimodal_fusion_dim: int
            Dimension of the multimodal fusion between the question encoding and
            the feature extraced from the image
        intermediate_dim: int
            output of the linear layer during used in the multimodal fusion
        prompt_tuner: PromptTuning
            will be used for generating prompt during inference
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
        prompt_tuner : PromptTuning,
        intermediate_dim=512):
        
        super(HadamardClassifier, self).__init__() 
        
        self.prompt_tuner = prompt_tuner
        
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
        
    def tune_question(self, question : str):
        return self.prompt_tuner.generate(question=question)[0]
        
    def forward(self, encoded_question, feature_vector): 
        
        v = self.prepare_multimodal_v(feature_vector)
        q = self.prepare_multimodal_q(encoded_question)
        
        h = torch.mul(v, q)
        
        logits = self.classifier(h)
        
        return logits
    
class ConcatClassifier(nn.Module):
    
    '''
    Classifier architecture that concatenates both outputs and pass the result through a non linear
    and a linear layer for classification.
    
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
        prompt_tuner : PromptTuning,
        intermediate_dim=512):
        
        super(ConcatClassifier, self).__init__() 
        
        self.prompt_tuner = prompt_tuner
        
        self.multimodal_fusion = nn.Sequential(
            nn.Linear(image_feature_dim + question_embedding_dim, intermediate_dim),
            nn.ReLU(),
            nn.Dropout(0.5)
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(intermediate_dim, vocabulary_size),
            nn.Sigmoid()
        )
        
    def tune_question(self, question : str):
        return self.prompt_tuner.generate(question=question)[0]
        
    def forward(self, encoded_question, feature_vector): 
        
        concat = torch.cat((encoded_question, feature_vector), dim=1)
        
        fusion = self.multimodal_fusion(concat)
        
        logits = self.classifier(fusion)
        
        return logits
    
class ConvVQA(nn.Module):
    def __init__(self, 
            vocabulary_size
        ):
        super(ConvVQA, self).__init__()
        
        # Prima convoluzione: in_channels=1 (per input 1D), out_channels=16
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm1d(16)
        self.dropout1 = nn.Dropout(0.3)  # Dropout dopo la prima convoluzione
        
        # Seconda convoluzione
        self.conv2 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm1d(32)
        self.dropout2 = nn.Dropout(0.3)  # Dropout dopo la seconda convoluzione
        
        # Terza convoluzione
        self.conv3 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm1d(64)
        self.dropout3 = nn.Dropout(0.3)  # Dropout dopo la terza convoluzione
        
        # Pooling globale per gestire l'input variabile
        self.global_pool = nn.AdaptiveAvgPool1d(1)

        # Strato finale di classificazione
        self.fc = nn.Linear(64, vocabulary_size)

    def forward(self, encoded_question, feature_vector):
        
        q = encoded_question / (encoded_question.norm(p=2) + 1e-8)
        v = feature_vector / (feature_vector.norm(p=2) + 1e-8)
        
        x = torch.cat((q, v), dim=1)
        
        # Aggiungi una dimensione per i canali: (batch_size, sequence_length) -> (batch_size, 1, sequence_length)
        x = x.unsqueeze(1)
        
        # Prima convoluzione con dropout
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.dropout1(x)
        
        # Seconda convoluzione con dropout
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.dropout2(x)
        
        # Terza convoluzione con dropout
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.dropout3(x)
        
        # Pooling globale
        x = self.global_pool(x)  # Output shape: (batch_size, 64, 1)
        x = x.squeeze(-1)        # Rimuove la dimensione 1 -> Output shape: (batch_size, 64)

        # Classificazione
        x = self.fc(x)           # Output shape: (batch_size, num_classes)
        return x
    
class BiggerConvVQA(nn.Module):
    def __init__(self, vocabulary_size):
        super(BiggerConvVQA, self).__init__()
        
        # Prima convoluzione: in_channels=1 (per input 1D), out_channels=16
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm1d(16)
        self.dropout1 = nn.Dropout(0.3)  # Dropout dopo la prima convoluzione
        
        # Seconda convoluzione
        self.conv2 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm1d(32)
        self.dropout2 = nn.Dropout(0.3)  # Dropout dopo la seconda convoluzione
        
        # Terza convoluzione
        self.conv3 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm1d(64)
        self.dropout3 = nn.Dropout(0.3)  # Dropout dopo la terza convoluzione
        
        # Quarta convoluzione
        self.conv4 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.bn4 = nn.BatchNorm1d(128)
        self.dropout4 = nn.Dropout(0.4)  # Dropout dopo la quarta convoluzione

        # Quinta convoluzione
        self.conv5 = nn.Conv1d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.bn5 = nn.BatchNorm1d(256)
        self.dropout5 = nn.Dropout(0.4)  # Dropout dopo la quinta convoluzione

        # Sesta convoluzione
        self.conv6 = nn.Conv1d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1)
        self.bn6 = nn.BatchNorm1d(512)
        self.dropout6 = nn.Dropout(0.4)  # Dropout dopo la sesta convoluzione
        
        # Settima convoluzione
        self.conv7 = nn.Conv1d(in_channels=512, out_channels=1024, kernel_size=3, stride=1, padding=1)
        self.bn7 = nn.BatchNorm1d(1024)
        self.dropout7 = nn.Dropout(0.5)  # Dropout dopo la settima convoluzione
        
        # Pooling globale per gestire l'input variabile
        self.global_pool = nn.AdaptiveAvgPool1d(1)

        # Strato finale di classificazione
        self.fc = nn.Linear(1024, vocabulary_size)

    def forward(self, encoded_question, feature_vector):
        
        q = encoded_question / (encoded_question.norm(p=2) + 1e-8)
        v = feature_vector / (feature_vector.norm(p=2) + 1e-8)
        
        x = torch.cat((q, v), dim=1)
        
        # Aggiungi una dimensione per i canali: (batch_size, sequence_length) -> (batch_size, 1, sequence_length)
        x = x.unsqueeze(1)
        
        # Prima convoluzione con dropout
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.dropout1(x)
        
        # Seconda convoluzione con dropout
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.dropout2(x)
        
        # Terza convoluzione con dropout
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.dropout3(x)

        # Quarta convoluzione con dropout
        x = F.relu(self.bn4(self.conv4(x)))
        x = self.dropout4(x)

        # Quinta convoluzione con dropout
        x = F.relu(self.bn5(self.conv5(x)))
        x = self.dropout5(x)

        # Sesta convoluzione con dropout
        x = F.relu(self.bn6(self.conv6(x)))
        x = self.dropout6(x)

        # Settima convoluzione con dropout
        x = F.relu(self.bn7(self.conv7(x)))
        x = self.dropout7(x)
        
        # Pooling globale
        x = self.global_pool(x)  # Output shape: (batch_size, 1024, 1)
        x = x.squeeze(-1)        # Rimuove la dimensione 1 -> Output shape: (batch_size, 1024)

        # Classificazione
        x = self.fc(x)           # Output shape: (batch_size, vocabulary_size)
        return x