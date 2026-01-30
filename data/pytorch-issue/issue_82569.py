import torch

x = torch.randn((50, 1000,1000),device='cuda')
torch.topk (x,10,-1) # Works
x = torch.randn((60, 1000,1000),device='cuda')
torch.topk (x,10,-1) # Crash