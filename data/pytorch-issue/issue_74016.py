import torch
from torch.utils.data import TensorDataset, DataLoader

print("Torch version:", torch.__version__)

x = torch.arange(12).reshape(6, 2)
ds = TensorDataset(x)
dl = DataLoader(ds, num_workers=8)

for x in dl:
    print(x)