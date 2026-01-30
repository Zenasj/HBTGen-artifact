import torch

a = torch.randn(3, 3) # CPU
with torch.cuda.amp.autocast(): # CUDA
    c = torch.mm(a, a)