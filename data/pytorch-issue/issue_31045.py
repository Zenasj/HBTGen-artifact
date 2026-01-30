import torch

targets = torch.empty(data.size(0)).random_(2).view(-1, 1)