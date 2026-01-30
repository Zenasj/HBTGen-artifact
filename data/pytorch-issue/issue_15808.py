import torch.nn as nn
import torchvision

import torch, torchvision
import torch_optimizer as optim

dataset_train = torchvision.datasets.FakeData(size = 10, num_classes = 3, transform = torchvision.transforms.ToTensor())
model     = torchvision.models.resnet18(num_classes = 3)
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr = 0.001)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 6, verbose = True)

loader_train = torch.utils.data.DataLoader(dataset_train, num_workers = 2)

for epoch in range(6):
    
    for i, (x, target) in enumerate(loader_train):
        
        # print(i)
        
        y_hat = model(x)
        loss = criterion(y_hat, target)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
    scheduler.step()

NCCL_P2P_DISABLE=1

OMP_NUM_THREADS=1 
MKL_NUM_THREADS=1