import torch
t = torch.randn(160, 256, 3, 1024).to("cuda")
t.topk(32, dim=-1)