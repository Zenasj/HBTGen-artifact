import torch
1 << torch.arange(10, device="mps")

tensor([ 0,  2,  4,  6,  8, 10, 12, 14, 16, 18], device='mps:0')

tensor([  1,   2,   4,   8,  16,  32,  64, 128, 256, 512], device='mps:0')