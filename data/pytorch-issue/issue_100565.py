import torch

with amp.autocast("cpu", dtype=torch.float32, enabled=False):
    x = torch.zeros(10)