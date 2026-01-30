import torchvision

import os, sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision import transforms
from joblib import Parallel, delayed

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = torch.nn.Linear(28 * 28, 10)

    def forward(self, x):
        return torch.relu(self.l1(x.view(x.size(0), -1)))

def main(num_workers, lr):
    data_set = MNIST(os.getcwd(), download=True, train=True, transform=transforms.ToTensor())
    data_loader = DataLoader(data_set, batch_size=64, num_workers=num_workers)
    model = Model().cuda()
    optim = torch.optim.Adam(model.parameters(), lr=lr)

    for batch in data_loader:
        x, y = batch
        y_hat = model(x.cuda())
        loss = F.cross_entropy(y_hat, y.cuda())
    print(f'completed {lr}')

if __name__ == '__main__':
    num_workers = int(sys.argv[1])
    lr_array = [1e-4, 1e-3, 1e-2]
    Parallel(n_jobs=3, backend='loky')(delayed(main)(num_workers, lr) for lr in lr_array)

import os, sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from joblib import Parallel, delayed
from joblib.externals.loky.backend.context import get_context


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = torch.nn.Linear(28 * 28, 10)

    def forward(self, x):
        return torch.relu(self.l1(x.view(x.size(0), -1)))

def main(num_workers, lr):
    X = torch.randn(20000, 28, 28)
    y = torch.randint(high=10, size=(20000,))
    data_set = torch.utils.data.TensorDataset(X, y)
    data_loader = DataLoader(data_set, batch_size=64, num_workers=num_workers,
                             multiprocessing_context=get_context('loky'))
    model = Model()
    optim = torch.optim.Adam(model.parameters(), lr=lr)

    for batch in data_loader:
        x, y = batch
        y_hat = model(x)
        loss = F.cross_entropy(y_hat, y)
    print(f'completed {lr}')

if __name__ == '__main__':
    num_workers = int(sys.argv[1])
    lr_array = [1e-4, 1e-3, 1e-2]
    Parallel(n_jobs=3, backend='loky')(delayed(main)(num_workers, lr) for lr in lr_array)