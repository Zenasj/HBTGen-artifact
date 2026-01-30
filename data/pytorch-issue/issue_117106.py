import torch

py
model = torch.compile(model, mode="reduce-overhead", fullgraph=True, dynamic=True)