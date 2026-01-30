import torch

from torch.utils.data import DataLoader
from torch.utils.data.sampler import Sampler
from torch.utils.data import Dataset
import numpy as np

print('running this script')

class Basic_Sampler(Sampler):
    def __init__(self, samples_list):
        self.sample_list = samples_list
        self.epoch_size = len(samples_list)
    def __iter__(self):
        return iter(self.sample_list)
    def __len__(self):
        return self.epoch_size

class fake_dataset(Dataset):
    def __init__(self):
            pass
    def __getitem__(self, item):
        return np.zeros(5)

sampler = Basic_Sampler([1,2,3])
dl = DataLoader(fake_dataset(), batch_size=2, shuffle=False, sampler=sampler, num_workers=1)

iter(dl).next()