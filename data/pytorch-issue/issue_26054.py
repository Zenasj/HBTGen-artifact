import torch.nn as nn

import torch
from torch.utils.data import Dataset, DataLoader

class RandomDataset(Dataset):
    def __init__(self):
        self.x = torch.rand(1, 2)
        
    def __getitem__(self, index):
        return self.x[index]

    def __len__(self):
        return self.x.size(0)

loader1 = DataLoader(RandomDataset(), num_workers=2, batch_size=1, pin_memory=True)
loader2 = DataLoader(RandomDataset(), num_workers=2, batch_size=1, pin_memory=True)

model = torch.nn.Linear(2, 4)
for data in loader1:
    print(model(data))

for data in loader2:
    print(model(data))