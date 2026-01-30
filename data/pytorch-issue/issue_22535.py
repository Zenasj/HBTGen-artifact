import random

import numpy as np
import torch
import torch.utils.data
import os
import re


PID = os.getpid()
STAT = f'/proc/{PID}/status'


def PRINT():
    with open(STAT, 'r') as f:
        print(re.search(r'Threads:[ \t]*\d+', f.read()).group(0))


class Dataset:

    def __len__(self):
        return 1000

    def __getitem__(self, index):
        x = np.random.randn(1, 512, 512)
        return torch.from_numpy(x)


dataset = Dataset()
loader = iter(torch.utils.data.DataLoader(
    dataset, batch_size=16, shuffle=True, num_workers=0, pin_memory=True, drop_last=True))


PRINT()

next(loader)

PRINT()