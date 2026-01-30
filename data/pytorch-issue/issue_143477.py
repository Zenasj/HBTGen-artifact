import torch
device = torch.device("mps")
mask_bool = torch.triu(torch.ones(1024, 1024, device=device), diagonal=1).bool()
attn_scores = torch.rand(48, 25, 1024, 1024, device=device)
attn_scores.masked_fill_(mask_bool, 0)

import torch

def foo(x, y):
   x.masked_fill_(y, 0)

device = torch.device("mps")
mask_bool = torch.triu(torch.ones(1024, 1024, device=device), diagonal=1).bool()
attn_scores = torch.rand(48, 25, 1024, 1024, device=device)

torch.compile(foo)(attn_scores, mask_bool)