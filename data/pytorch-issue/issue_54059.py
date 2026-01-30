import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim

import torch.multiprocessing as mp
import torch.distributed as dist

import numpy as np

from torchvision import datasets
import torchvision.transforms as transforms


class img_classifier(nn.Module):
    def __init__(self):
        super(img_classifier, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 4 * 4, 500)
        self.fc2 = nn.Linear(500, 10)
        self.dropout = nn.Dropout(0.2)


    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(-1, 64 * 4 * 4)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))        

        return x

def train(gpu, args):
    
    rank = args['nr'] * args['gpus'] + gpu                          
    dist.init_process_group(                                   
        backend='nccl',                                         
        init_method='env://',                                   
        world_size=args['world_size'],                              
        rank=rank                                               
    )
    
    torch.cuda.set_device(gpu)
    
    ## Dataset
    transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    train_data = datasets.CIFAR10('CIFAR10_data/', train=True, download=True, transform=transform)
    valid_data = datasets.CIFAR10('CIFAR10_data/', train=False, download=True, transform=transform)
    
    train_sampler = torch.utils.data.distributed.DistributedSampler(
    	train_data,
    	num_replicas=args['world_size'],
    	rank=rank,
    )
    
    batch_size = len(train_data)//args['world_size']
    
    if gpu == 0:
        print(batch_size)

    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, sampler=train_sampler)
    valid_loader = torch.utils.data.DataLoader(valid_data, batch_size=batch_size)
    
    ##Model
    
    model = img_classifier().cuda(gpu)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.02)
    
    model = nn.parallel.DistributedDataParallel(model, device_ids=[gpu])
    
    n_epochs = 20
    valid_loss_min = np.Inf

    for epoch in range(1, n_epochs+1):

        train_loss = 0.0
        valid_loss = 0.0

        model.train()
               
        for data, target in train_loader:
            if torch.cuda.is_available:
                data, target = data.cuda(), target.cuda()
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()*data.size(0)


        # validate the model
        if gpu==0 :
            model.eval()
            for data, target in valid_loader:
                if torch.cuda.is_available:
                    data, target = data.cuda(), target.cuda()
                output = model(data)
                loss = criterion(output, target)
                valid_loss += loss.item()*data.size(0)

            # calculate average losses
            train_loss = train_loss/len(train_loader.dataset)
            valid_loss = valid_loss/len(valid_loader.dataset)

            # print training/validation statistics 
            print('Epoch: {} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f}'.format(
                epoch, train_loss, valid_loss))

            # save model if validation loss has decreased
            if valid_loss <= valid_loss_min:
                print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(
                valid_loss_min,
                valid_loss))
                torch.save(model.state_dict(), 'model_cifar.pt')
                valid_loss_min = valid_loss

        dist.barrier()
        
def main():
    args = {
        'gpus' : 2,
        'nodes' : 1,
        'nr': 0
    }
    
    args['world_size'] = args['gpus'] * args['nodes']
    mp.spawn(train, nprocs=args['gpus'], args=(args,))
    
    
if __name__ == '__main__':
    main()

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim

import torch.multiprocessing as mp
import torch.distributed as dist

import numpy as np

from torchvision import datasets
import torchvision.transforms as transforms

import os
os.environ["MASTER_ADDR"] = "localhost"
os.environ["MASTER_PORT"] = "12346"
os.environ["NCCL_DEBUG"] = "INFO"
os.environ["NCCL_DEBUG_SUBSYS"] = "ALL"

class img_classifier(nn.Module):
    def __init__(self):
        super(img_classifier, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 4 * 4, 500)
        self.fc2 = nn.Linear(500, 10)
        self.dropout = nn.Dropout(0.2)


    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(-1, 64 * 4 * 4)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))        

        return x

def train(gpu, args):
    
    rank = args['nr'] * args['gpus'] + gpu                          
    dist.init_process_group(                                   
        backend='nccl',                                         
        init_method='env://',                                   
        world_size=args['world_size'],                              
        rank=rank                                               
    )
    
    torch.cuda.set_device(gpu)
    
    ## Dataset
    transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    train_data = datasets.CIFAR10('CIFAR10_data/', train=True, download=True, transform=transform)
    valid_data = datasets.CIFAR10('CIFAR10_data/', train=False, download=True, transform=transform)
    
    train_sampler = torch.utils.data.distributed.DistributedSampler(
    	train_data,
    	num_replicas=args['world_size'],
    	rank=rank,
    )
    
    batch_size = len(train_data)//args['world_size']
    
    if gpu == 0:
        print(batch_size)

    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, sampler=train_sampler)
    valid_loader = torch.utils.data.DataLoader(valid_data, batch_size=batch_size)
    
    ##Model
    
    model = img_classifier().cuda(gpu)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.02)
    
    model = nn.parallel.DistributedDataParallel(model, device_ids=[gpu])
    
    n_epochs = 20
    valid_loss_min = np.Inf

    for epoch in range(1, n_epochs+1):

        train_loss = 0.0
        valid_loss = 0.0

        model.train()
               
        for data, target in train_loader:
            if torch.cuda.is_available:
                data, target = data.cuda(), target.cuda()
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()*data.size(0)

        dist.barrier()

        # validate the model
        if gpu==0 :
            model.eval()
            for data, target in valid_loader:
                if torch.cuda.is_available:
                    data, target = data.cuda(), target.cuda()
                output = model(data)
                loss = criterion(output, target)
                valid_loss += loss.item()*data.size(0)

            # calculate average losses
            train_loss = train_loss/len(train_loader.dataset)
            valid_loss = valid_loss/len(valid_loader.dataset)

            # print training/validation statistics 
            print('Epoch: {} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f}'.format(
                epoch, train_loss, valid_loss))

            # save model if validation loss has decreased
            if valid_loss <= valid_loss_min:
                print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(
                valid_loss_min,
                valid_loss))
                torch.save(model.state_dict(), 'model_cifar.pt')
                valid_loss_min = valid_loss


def main():
    args = {
        'gpus' : 2,
        'nodes' : 1,
        'nr': 0
    }
    
    args['world_size'] = args['gpus'] * args['nodes']
    mp.spawn(train, nprocs=args['gpus'], args=(args,))
    
    
if __name__ == '__main__':
    main()