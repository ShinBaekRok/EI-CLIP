import torch
from torchvision import transforms
from torch import nn
import numpy as np
import glob
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import  torch.nn.functional as F
import cv2

class Image_Encoding(nn.Module):
    def __init__(self, backbone, num_labels):
        super().__init__()
        self.ViT = backbone
        self.Act = nn.ReLU(True)
        self.encoding = nn.Linear(1000, 40)
        
    def forward(self, x):
        encoded_output = self.ViT(x)
        embedding = self.Act(encoded_output)
        classification_output = self.encoding(encoded_output)
        
        return embedding, classification_output