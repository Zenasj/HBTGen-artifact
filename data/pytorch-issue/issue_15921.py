import torch

print(torch.fmod(torch.tensor(421123111).cpu(), 2))  # 1
print(torch.fmod(torch.tensor(421123111).cuda(), 2))  # 0