import torch.nn.functional as F
import random

import torchvision.transforms.functional as TF
import torch
import numpy as np
from torch.utils.data import DataLoader, Dataset

class ExampleDataset(Dataset):
    def __init__(self):
        self.data = [np.random.rand(3, 224, 224) for _ in range(100)]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return TF.to_tensor(self.data[idx])

dataset = ExampleDataset()
dataloader = DataLoader(dataset, batch_size=32)

for data in dataloader:
    print(data.shape)
    break