import torch.nn as nn

from torch.utils.data import DataLoader, Dataset
import torch
from torch import nn
from torch.nn.attention.flex_attention import flex_attention, create_block_mask, _create_empty_block_mask, BlockMask

# flex_attention = torch.compile(flex_attention)

# Create a simple custom dataset
class MyDataset(Dataset):
    def __init__(self):
        self.data = torch.randn(100, 32)
        self.labels = torch.randint(0, 2, (100,)) 

    def __len__(self):
        return 10000000

    def __getitem__(self, idx):
        return self.data[idx % 100], self.labels[idx % 100]

class MyModel(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.loss = torch.nn.MSELoss()
        self.q_proj = nn.LazyLinear(128)
        self.k_proj = nn.LazyLinear(128)
        self.v_proj = nn.LazyLinear(128)

    def forward(self, x, y):
        x = x[None, None, :, :]
        r = x
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        r = flex_attention(q, k, v)
        return self.loss(torch.sum(r[0, 0, :, :], dim=-1), y)


dataset = MyDataset()
data_loader = DataLoader(dataset, batch_size=3, shuffle=True)

model = MyModel()
model.compile()
for x, y in data_loader:
    print(model(x, y))
    break