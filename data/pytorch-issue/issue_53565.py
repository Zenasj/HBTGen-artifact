import torch.nn as nn

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


class MyDataset(Dataset):
    def __init__(self):
        self[0]  # The important thing is that conv1d is called here

    def __getitem__(self, index):
        x = torch.Tensor(1, 1, 24000)  # Needs to be long enough
        x = F.conv1d(x, torch.ones(1, 1, 2))  # Causes segfault
        return x

    def __len__(self):
        return 1


# num_workers>0 necessary to reproduce error
loader = DataLoader(MyDataset(), num_workers=1)
for x in loader:
    pass