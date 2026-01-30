import torch.nn as nn

import torch
import torch.nn.functional as F

device = "cuda:0"
for use_grad in [True, False]:
    bag = torch.nn.EmbeddingBag(256, 256, mode="sum", device=device)
    bag.requires_grad_(use_grad)
    x = torch.arange(1, 5, device=device).expand(3, -1)
    w = torch.rand(3, 4, device=device, requires_grad=True)
    bag(x, per_sample_weights=F.softmax(w, dim=-1))
    print(f"Works when bag.weight.requires_grad({use_grad})")

import torch
import torch.nn.functional as F

device = "cuda:0"
for use_grad in [True, False]:
    bag = torch.nn.EmbeddingBag(256, 256, mode="sum", device=device)
    bag.requires_grad_(use_grad)
    x = torch.arange(1, 5, device=device).expand(3, -1)
    w = torch.rand(3, 4, device=device)  # , requires_grad=True)
    bag(x, per_sample_weights=F.softmax(w, dim=-1))
    print(f"Works when bag.weight.requires_grad({use_grad})")