import torch

torch.multinomial(torch.FloatTensor([0, 1, 0, 0]), 3, replacement=False)