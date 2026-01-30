import torch
dummy = torch.randn(1, 512, 1024, 4, device='cuda:0')*1000
inp = dummy[..., :3]
out=torch.mean(inp)