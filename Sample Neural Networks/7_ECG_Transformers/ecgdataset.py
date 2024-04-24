import numpy as np
import pandas as pd 
import torch 
from torch.utils.data import Dataset, DataLoader 
from sklearn.preprocessing import OneHotEncoder 

from sklearn.model_selection import train_test_split 

class ECGDataset(Dataset): 
    def __init__(self,x,y):
        super().__init__()
        self.X = torch.tensor(x)
        self.Y = torch.tensor(y)
    
    def __len__(self):
        return len(self.Y)
    
    def __getitem__(self, index):
        return self.X[index].unsqueeze(1), self.Y[index]
    
