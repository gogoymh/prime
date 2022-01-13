import torch
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
import torch.nn as nn
import torch.optim as optim

from network8_2 import net
from utils import WarmupLinearSchedule, SAM

train_loader = DataLoader(
                datasets.CIFAR10(
                        "../data/CIFAR10",
                        train=True,
                        download=True,
                        transform=transforms.Compose([
                                transforms.RandomCrop(32, padding=4),
                                transforms.RandomHorizontalFlip(),
                                transforms.ToTensor(),
                                transforms.Normalize(
                                        (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))]
                                ),
                        ),
                batch_size=128, shuffle=True)#, pin_memory=True)


test_loader = DataLoader(
                datasets.CIFAR10(
                        '../data/CIFAR10',
                        train=False,
                        download=True,
                        transform=transforms.Compose([
                                transforms.ToTensor(),
                                transforms.Normalize(
                                        (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))]
                                ),
                        ),
                batch_size=1, shuffle=False)#, pin_memory=True)


device = torch.device("cuda:0")
model = net().to(device)

#params = list(model.parameters()) + list(init.parameters())
#optimizer = optim.Adam(params, lr=0.01)
#optimizer = optim.Adam(model.parameters(), lr=1e-3)#, weight_decay=0.1)
#optimizer = optim.SGD(model.parameters(), lr=3e-2, momentum=0.9, weight_decay=0)
criterion = nn.CrossEntropyLoss()

#scheduler = WarmupLinearSchedule(optimizer, warmup_steps=15, t_total=300)

base_optimizer = torch.optim.SGD  
optimizer = SAM(model.parameters(), base_optimizer, lr=3e-2, momentum=0.9)    
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size = 5, gamma = 0.75)

best_acc = 0
for epoch in range(300):
    runnning_loss = 0
    model.train()
    for x, y in train_loader:
        optimizer.zero_grad()
               
        #output, halt = model(x.float().to(device))
        #print((torch.isnan(output)).sum())
        #loss = criterion(output, y.long().to(device))/halt.mean() + halt.mean()
        loss = model(x.float().to(device), y.unsqueeze(1).long().to(device))
        loss.backward()
        optimizer.first_step(zero_grad=True)
        
        model(x.float().to(device), y.unsqueeze(1).long().to(device)).backward()
        optimizer.second_step(zero_grad=True)
        
        #optimizer.step()
        runnning_loss += loss.item()
        #print(loss.item(), halt.mean().item())
    scheduler.step()
    
    runnning_loss /= len(train_loader)
    print("[Epoch:%d] [Loss:%f]" % ((epoch+1), runnning_loss), end=" ")
    
    
    with torch.no_grad():
        model.eval()
        #print("[Scale:%f]" % model.scale, end=" ")
        correct1 = 0
        correct2 = 0
        #mean_cnt = 0
        for x, y in test_loader:
            output1, output2 = model(x.float().to(device), y.long().to(device))
            pred1 = output1.argmax(1, keepdim=True)
            correct1 += pred1.eq(y.long().to(device).view_as(pred1)).sum().item()
            pred2 = output2.argmax(1, keepdim=True)
            correct2 += pred2.eq(y.long().to(device).view_as(pred2)).sum().item()
            #mean_cnt += cnt    
            
        accuracy1 = correct1 / len(test_loader.dataset)
        accuracy2 = correct2 / len(test_loader.dataset)
        #mean_cnt = mean_cnt / len(test_loader.dataset)
        
        if max(accuracy1, accuracy2) >= best_acc:
            print("[Last Accuracy:%f] [Ensemble Accuracy:%f] **Best**" % (accuracy1, accuracy2))
            best_acc = max(accuracy1, accuracy2)
        else:
            print("[Last Accuracy:%f] [Ensemble Accuracy:%f]" % (accuracy1, accuracy2))

        
        
    #scheduler.step()
    
    #torch.save({'model_state_dict': model.state_dict()}, "/data/ymh/gpu_test/resnet56_preact.pth")