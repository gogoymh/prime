import numpy as np
import torch
from torch.utils.data import DataLoader# TensorDataset
from torchvision import transforms, datasets
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import math
from einops import rearrange, repeat


class BasicBlock(nn.Module):
    def __init__(self, planes):
        super(BasicBlock, self).__init__()
        
        self.relu = nn.LeakyReLU()
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv1 = nn.Conv2d(planes, planes, 3, 1, 1, bias=False)
        
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, 3, 1, 1, bias=False)

    def forward(self, x):
        
        out = self.bn1(x)
        out = self.relu(out)
        out = self.conv1(out)
        
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv2(out)

        return out + x




class net(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.conv1 = nn.Conv2d(3, 16, 3, 1, 1, bias=False)
        self.module1 = SkipNet(16, 9)
        self.down1 = nn.Conv2d(16, 32, 3, 2, 1, bias=False)
        self.module2 = SkipNet(32, 9)
        self.down2 = nn.Conv2d(32, 64, 3, 2, 1, bias=False)
        self.module3 = SkipNet(64, 9)
        
        self.max = 27
        self.desired = 27
        
    def forward(self, x, y):
        b = x.shape[0]
        x = self.conv1(x)
        
        if self.training:
            out1, halt1, x = self.module1(x) # (B, max, class), (B, max)
            x = self.down1(x)
            out2, halt2, x = self.module2(x)
            x = self.down2(x)
            out3, halt3, x = self.module3(x)
            
            out = torch.cat((out1,out2,out3), dim=1)
            halt = torch.cat((halt1,halt2,halt3), dim=1)
            
            labels = repeat(y, 'b n -> b (l n)', l = self.max) # (B, 1) -> (B, max)
            logits = rearrange(out, 'b l d -> b d l') # (B, max, class) -> (B, class, max)
            ce_loss = F.cross_entropy(logits, labels, reduction='none')

            min_ce_loss = ce_loss.min(dim=1)[0]

            index = ce_loss.argmin(1, keepdim=False) + 1
            
            ones = torch.ones((b, self.max)).to(x.device)
            halt_labels = list(map(lambda i: change(ones,i), enumerate(index)))
            halt_logits = list(map(lambda i: halt[i[0], :i[1]], enumerate(index)))
            halt_ce_loss = torch.stack(list(map(lambda x: F.binary_cross_entropy(x[0], x[1], reduction='mean'), zip(halt_logits, halt_labels))))
            
            abs_halt = torch.stack(list(map(lambda x: x.sum(), halt_logits)))            

            #loss = ce_loss.mean() + halt_ce_loss.mean() + (1/self.desired) * abs_halt.mean() + min_ce_loss.mean()
            loss = halt_ce_loss.mean() + (1/self.desired) * abs_halt.mean() + min_ce_loss.mean()
            
            return loss.mean()
        
        else:
            out, cnt, go, x = self.module1(x)
            if go:
                x = self.down1(x)
                out, cnt, go, x = self.module2(x)
                cnt += 9
                if go:
                    x = self.down2(x)
                    out, cnt, _, _ = self.module3(x)
                    cnt += 18
            return out, cnt

def change(tensor, x):
    tensor[x[0],x[1]-1] = 0
    return tensor[x[0], :x[1]]

class SkipNet(nn.Module):
    def __init__(self, planes, budget):
        super().__init__()
        
        self.max = budget
        self.module = nn.ModuleList()
        for _ in range(self.max):
            self.module.append(BasicBlock(planes)) # each layer has different parameters
            
        self.classifier = nn.ModuleList()
        for _ in range(self.max):
            self.classifier.append(
                nn.Sequential(
                    nn.AdaptiveAvgPool2d((1, 1)),
                    nn.Flatten(),
                    nn.Linear(planes, 10)
                    ))
        
        self.halt = nn.ModuleList()
        for _ in range(self.max):
            self.halt.append(nn.Sequential(
                nn.AdaptiveAvgPool2d((1, 1)),
                nn.Flatten(),
                nn.Linear(planes,1),
                nn.Sigmoid()
                ))
    
    def forward(self, x):
        outputs = []
        halts = []
        
        if self.training:
            for i in range(self.max):
                x = self.module[i](x)
                
                output = self.classifier[i](x)
                
                halt = self.halt[i](x).squeeze()
                
                outputs.append(output)
                halts.append(halt)
            
            outputs = torch.stack(outputs, dim = 1)
            halts = torch.stack(halts, dim = 1)
            return outputs, halts, x
        else:
            x = self.module[0](x)
            for i in range(self.max):
                output = self.classifier[i](x) 
                
                halt = self.halt[i](x).squeeze(1)
                
                check = (halt >= 0.5)
                if check.sum() == 0 or (i+1) == self.max:
                    return output, (i+1), check, x
                else:
                    x[check] = self.module[i+1](x[check])
    

def roop2(var):
    print(var, 'input')
    var = var + 0.1
    
    check = var < 0.5
    if check.sum() == 0:
        print(var, 'No F')
        return var
    else:
        var[check] = roop2(var[check])
        print(var, 'Re check')
        return var 
    
if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    B = 3
    a = torch.randn((B,3,32,32)).to(device).softmax(dim=-1)
    target = torch.randint(10, (B,1), dtype=torch.int64).to(device)
    b = net().to(device)
    
    b.train()
    loss = b(a, target)
    print(loss)
    
    c = torch.randn((1,3,32,32)).to(device).softmax(dim=-1)
    b.eval()
    out1, out2 = b(c, target)
    print(out1.shape, out2)
