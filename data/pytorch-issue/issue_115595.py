import torch

torch.signal.windows.kaiser(3, beta=7.8)  # tensor([nan, 1., nan])

torch.sqrt(beta * beta - torch.pow(k, 2))