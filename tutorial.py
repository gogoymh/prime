from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch
import pandas as pd
import numpy as np
    
class Numberset(Dataset):
    def __init__(self, N, path = '2T_part1.txt'):
        super().__init__()
        
        self.data = pd.read_csv(path, header = None, delimiter = "\t")
        x = N // 10
        y = (N-1) % 10
        self.S = len(str(self.data.iloc[x,y]))
        self.len = self.S * N
        
    def __getitem__(self, total_index):
        
        index = total_index // self.S
        place = total_index % self.S
        
        num_class, exist = self.gen(index, place)
        
        return torch.FloatTensor([index, place]), num_class, exist
    
    def __len__(self):
        return self.len
    
    def gen(self, index, place):
        x = index // 10
        y = index % 10
        
        value = str(self.data.iloc[x,y])
        standard = len(value)-1
        if standard < place:
            num_class = 10
            exist = 1
        elif standard == place:
            num_class = value[-(place+1)] 
            exist = 1
        else:
            num_class = value[-(place+1)] 
            exist = 0
        
        return int(num_class), exist

if __name__ == "__main__":
    a = Numberset(28)
    
    index = 83
    b, c, d = a.__getitem__(index)
    print(b, c, d)
    '''
    loader = DataLoader(dataset=a, batch_size=4, shuffle = True)
    
    d, e = loader.__iter__().next()
    print(d)
    print(d.shape)
    print(e)
    print(e.shape)
    '''
