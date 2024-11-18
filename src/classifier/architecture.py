import sys
sys.path.append('src')

import torch
import torch.nn as nn

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
        
        super(HadamardClassifier, self).__init__() 
        
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
        intermediate_dim=512):
        
        super(ConcatClassifier, self).__init__() 
        
        self.multimodal_fusion = nn.Sequential(
            nn.Linear(image_feature_dim + question_embedding_dim, intermediate_dim),
            nn.ReLU(),
            nn.Dropout(0.5)
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(intermediate_dim, vocabulary_size),
            nn.Sigmoid()
        )
        
    def forward(self, encoded_question, feature_vector): 
        
        concat = torch.cat((encoded_question, feature_vector), dim=1)
        
        fusion = self.multimodal_fusion(concat)
        
        logits = self.classifier(fusion)
        
        return logits