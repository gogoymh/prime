import torch
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
import torch.nn as nn
import torch.optim as optim
import numpy as np

from tutorial import Numberset
from network1 import net

numofset = 100000
train_ratio = 0.8

wholeset = set([i for i in range(numofset)])
train_idx = np.random.choice(numofset, int(numofset*train_ratio), replace=False).tolist()
val_idx = list(wholeset - set(train_idx))

Primeset = Numberset(numofset)
train_sampler = SubsetRandomSampler(train_idx)
train_loader = DataLoader(dataset=Primeset, batch_size=128, sampler=train_sampler)
valid_sampler = SubsetRandomSampler(val_idx)
valid_loader = DataLoader(dataset=Primeset, batch_size=128, sampler=valid_sampler)

device = torch.device("cuda:0")
model = net().to(device)

optimizer = optim.Adam(model.parameters(), lr=0.001)
#optimizer = optim.SGD(model.parameters(), lr=0.1)
criterion = nn.CrossEntropyLoss()

best_acc = 0
for epoch in range(300):
    runnning_loss = 0
    for x, y, _ in train_loader:
        optimizer.zero_grad()
               
        #output1, output2 = model(x.float().to(device))
        output1, _ = model(x.float().to(device))
        #print((torch.isnan(output1)).sum(), (torch.isnan(output2)).sum())
        loss = criterion(output1, y.long().to(device))# + criterion(output2, z.long().to(device))
        loss.backward()
        optimizer.step()
        runnning_loss += loss.item()
        #print(loss.item())
        
    runnning_loss /= len(train_loader)
    print("[Epoch:%d] [Loss:%f]" % ((epoch+1), runnning_loss), end=" ")
    
    
    accuracy1 = 0
    accuracy2 = 0
    with torch.no_grad():
        model.eval()
        #print("[Scale:%f]" % model.scale, end=" ")
        correct1 = 0
        '''
        TP = 0
        TN = 0
        FP = 0
        FN = 0
        '''
        for x, y, _ in valid_loader:
            #output1, output2 = model(x.float().to(device))
            output1, _ = model(x.float().to(device))
            pred1 = output1.argmax(1, keepdim=True)
            correct1 += pred1.eq(y.long().to(device).view_as(pred1)).sum().item()
            '''
            pred2 = output2.argmax(1, keepdim=True)
            pred2 = pred2.view(-1)
                    
            Trues = pred2[pred2 == z.long().to(device).view(-1)]
            Falses = pred2[pred2 != z.long().to(device).view(-1)]
                
            TP += (Trues == 1).sum().item()
            TN += (Trues == 0).sum().item()
            FP += (Falses == 1).sum().item()
            FN += (Falses == 0).sum().item()
            '''
        '''
        accuracy2 = (TP + TN)/(TP + TN + FP + FN)
        if TP == 0:
            precision = 0
            recall = 0
            dice = 0
        else:
            precision = TP / (TP + FP)
            recall = TP / (TP + FN)
            dice = (2 * TP) / ((2 * TP) + FP + FN)
        '''  
            
        accuracy1 = correct1 / len(valid_loader.dataset)

        if accuracy1 >= best_acc:
            print("[Accuracy1:%f] **Best**" % accuracy1)
            best_acc = accuracy1
        else:
            print("[Accuracy1:%f]" % accuracy1)
        '''
        print("[Accuracy2:%f]" % accuracy2, end=" ")
        print("[Precision:%f]" % precision, end=" ")
        print("[Recall:%f]" % recall, end=" ")
        print("[F1 score:%f]" % dice)
        '''
        print("="*100)
        model.train()