import torch
print(torch.ones(128, dtype=torch.bool, device="mps").cumsum(0)[-1])
# tensor(-128, device='mps:0')