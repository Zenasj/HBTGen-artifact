import torch

print(torch.__version__)
torch.max(torch.tensor(1.0, dtype=torch.half))