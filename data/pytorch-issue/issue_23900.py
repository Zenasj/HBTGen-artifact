import torch

from torch.utils.data import Dataset
import itertools
import numpy as np
import resource

def print_memusage():
    memuse = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    print(f'Usage: {memuse / 1e6:0.3f} GB')

class TestDataset(Dataset):
    def __getitem__(self, index):
        return np.arange(1000 * 1000, dtype='u8')
    def __len__(self):
        return 1000000

dataset = TestDataset()
infinite_iterator = itertools.cycle(dataset)

if True:  # change to False to run infinite iterator
    print('Normal Dataset')
    for _ in zip(range(20), dataset):
        print_memusage()
else:
    print('Infinite Iterator')
    for _ in zip(range(20), infinite_iterator):
        print_memusage()

def cycle(iterable):
    iterator = iter(iterable)
    while True:
        try:
            yield next(iterator)
        except StopIteration:
            iterator = iter(iterable)