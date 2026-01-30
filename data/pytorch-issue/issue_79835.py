import torch

3
torch.arange(10, device="mps", dtype=torch.int32) + torch.tensor(5).to('mps', dtype=torch.int32)