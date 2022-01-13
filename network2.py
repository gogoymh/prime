import numpy as np
import torch
from torch.utils.data import DataLoader# TensorDataset
from torchvision import transforms, datasets
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import math

class RepeatBlock(nn.Module):
    def __init__(self, planes):
        super().__init__()
        
        self.relu = nn.LeakyReLU()
        
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv1 = nn.Conv2d(planes, planes, 3, 1, 1)
        
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, 3, 1, 1)
        
        self.dropout = nn.Dropout2d(p=0.1)
        
    def forward(self, x):
        identity = x
        
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv1(x)
        
        x = self.bn2(x)
        x = self.relu(x)
        x = self.conv2(x)

        return self.dropout(x) + identity


class net(nn.Module):
    def __init__(self, planes):
        super().__init__()
        
        self.module = RepeatBlock(planes)

        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Sequential(
            nn.LayerNorm(planes+1),
            nn.Linear(planes+1,16),
            nn.LeakyReLU(),
            nn.LayerNorm(16),
            nn.Linear(16,10)
            )
        self.halt = nn.Sequential(
            nn.LayerNorm(planes+1),
            nn.Linear(planes+1,16),
            nn.LeakyReLU(),
            nn.LayerNorm(16),
            nn.Linear(16,1),
            nn.Sigmoid()
            )
    
    def forward(self, x, cnt, halt, accum):
        x = self.module(x)
        
        out = torch.cat((torch.flatten(self.pool(x),1), (accum/100.).unsqueeze(1)), dim=1)
        y = self.halt(out).squeeze(1)
        which = y >= 0.5
        
        if which.sum() == 0 or cnt == 100:
            if self.training:
                return self.fc(out), (halt+y).mean()
            else:
                return self.fc(out), cnt
        else:
            x[which] = self.forward(x[which], cnt+1, halt+y, accum+which)
            return x
    
    '''
    def forward(self, x, cnt, halt, accum):
        out = torch.cat((torch.flatten(self.pool(x),1), (accum/100.).unsqueeze(1)), dim=1)
        y = self.halt(out).squeeze(1)
        which = y >= 0.5
        
        if which.sum() == 0 or cnt == 100:
            if self.training:
                return self.fc(out), halt.mean()
            else:
                return self.fc(out), cnt
        else:
            x[which] = self.module(x[which])
            x = self.forward(x, cnt+1, halt+y, accum+which)
            return x
    '''

def roop(var):
    check = var < 0.5
    
    if check.sum() == 0:
        return var
    else:
        var[check] = var[check] + 0.1
        print(var)
        var = roop(var)
        return var
    
if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    B = 3
    a = torch.randn((B,3,10,10)).to(device).softmax(dim=-1)
    b = net(3).to(device)
    
    c, d = b(a, 0, torch.zeros(B).to(device), torch.zeros(B).to(device))
    print(c.shape)
    print(d.item())
