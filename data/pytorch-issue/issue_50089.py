import torch
import time
import multiprocessing as mp
from torch.nn.functional import mse_loss
from torch.utils.data import TensorDataset, DataLoader, Dataset
import torch.nn as nn

class DatasetFilter(Dataset):
    def __init__(self, data):
        self.X = data[0]
        self.y = data[1]
        self.goodloc = torch.arange(start=0, end=1000, step=5)
        self.len = len(self.goodloc)
        #self.ddd = set([1,2,3]) #remove the error; this line is inserted just for showing error
        self.ddd = set(torch.tensor([1,2,3])) #cause the error; this line is inserted just for showing error

    def __getitem__(self, item):
        return (
            self.X[self.goodloc[item]],
            self.y[self.goodloc[item]])

    def __len__(self):
        return self.len

def worker(data_share):
    ds = DatasetFilter(data_share)
    train_dataloader = DataLoader(
            ds,
            batch_size=100,
            shuffle=True,
            drop_last=True,
            num_workers=4,
            pin_memory=True)

    model = nn.Linear(in_features=data_share[0].shape[1], out_features=1)
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3, momentum=0.9)

    for epoch in range(10):
        for x, y in train_dataloader:
            model.train()
            optimizer.zero_grad()
            output = model(x)
            loss = mse_loss(output.squeeze(), y)
            loss.backward()
            optimizer.step()

if __name__ == '__main__':
    share_resource = (torch.rand(1000, 3), torch.rand(1000))

    mp.set_start_method('spawn')
    nprocs = 2
    processes = []
    for rank in range(nprocs):
        p = mp.Process(target=worker, args=(share_resource,))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

import torch
import time
import torch.multiprocessing as mp
from torch.nn.functional import mse_loss
from torch.utils.data import TensorDataset, DataLoader, Dataset
import torch.nn as nn

class DatasetFilter(Dataset):
    def __init__(self, data):
        self.X = data[0]
        self.y = data[1]
        self.goodloc = torch.arange(start=0, end=1000, step=5)
        self.len = len(self.goodloc)
        #self.ddd = set([1,2,3]) #remove the error; this line is inserted just for showing error
        self.ddd = set(torch.tensor([1,2,3])) #cause the error; this line is inserted just for showing error

    def __getitem__(self, item):
        return (
            self.X[self.goodloc[item]],
            self.y[self.goodloc[item]])

    def __len__(self):
        return self.len

def worker(data_share):
    ds = DatasetFilter(data_share)
    train_dataloader = DataLoader(
            ds,
            batch_size=100,
            shuffle=True,
            drop_last=True,
            num_workers=4,
            pin_memory=True)

    model = nn.Linear(in_features=data_share[0].shape[1], out_features=1)
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3, momentum=0.9)

    for epoch in range(10):
        for x, y in train_dataloader:
            model.train()
            optimizer.zero_grad()
            output = model(x)
            loss = mse_loss(output.squeeze(), y)
            loss.backward()
            optimizer.step()

def main_worker(rank, *args):
    share_memory_resources = args
    worker(share_memory_resources)

if __name__ == '__main__':
    share_resource = (torch.rand(1000, 3), torch.rand(1000))

    mp.spawn(main_worker, args=(share_resource), nprocs=2, join=True)

    #mp.set_start_method('spawn')
    #nprocs = 2
    #processes = []
    #for rank in range(nprocs):
    #    p = mp.Process(target=worker, args=(share_resource,))
    #    p.start()
    #    processes.append(p)

    #for p in processes:
    #    p.join()