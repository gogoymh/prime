import numpy as np
import torch
from torch.utils.data import DataLoader# TensorDataset
from torchvision import transforms, datasets
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import math

class net(nn.Module):
    def __init__(self, node = 1000, layer = 100):
        super().__init__()
        
        self.init = nn.Linear(2,node)
        self.module = nn.ModuleList()
        for i in range(layer):
            self.module.append(nn.Sequential(
                nn.LayerNorm(node),
                nn.ReLU(inplace=True),
                nn.Linear(node,node),
                nn.Dropout(p=0.1)
                ))
        
        self.out1 = nn.Linear(node, 11)
        self.out2 = nn.Linear(node, 2)
        
    def forward(self, x):
        x = self.init(x)
        for operation in self.module:
            x = x + operation(x)
        out1 = self.out1(x)
        out2 = self.out2(x)
        return out1, out2
    
if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    a = torch.randn((3,2)).to(device)
    b = net(10, 5).to(device)
    
    c, d = b(a)
    print(c.shape, d.shape)