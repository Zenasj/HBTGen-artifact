import random

import cupy as cp
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

class TestDataset(Dataset):
    def __init__(self):
        pass

    def __getitem__(self, index):
        # cupy code here!
        cp.random.seed(1)
        return index

    def __len__(self):
        return 1000

def test_worker(gpu=0):
    dataset = TestDataset()

    train_loader = DataLoader(dataset,
                              1,
                              shuffle=True,
                              pin_memory=True,
                              num_workers=1)

    for index in train_loader:
        print(index)

if __name__ == '__main__':
    test_worker()

import cupy as cp
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch.multiprocessing as mp


class TestDataset(Dataset):
    def __init__(self):
        pass

    def __getitem__(self, index):
        # cupy code here!
        cp.random.seed(1)
        return index

    def __len__(self):
        return 1000

def test_worker(gpu=0):
    dataset = TestDataset()

    train_loader = DataLoader(dataset,
                              1,
                              shuffle=True,
                              pin_memory=True,
                              num_workers=1)

    for index in train_loader:
        print(index)

if __name__ == '__main__':
    mp.spawn(
        test_worker
    )