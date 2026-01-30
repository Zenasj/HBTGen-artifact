import torch
import torch.nn as nn

inp = torch.randn(190,64,64, 2933, device="cuda")
pool = nn.MaxPool2d(2, 2)
out=pool(inp)
torch.cuda.synchronize()