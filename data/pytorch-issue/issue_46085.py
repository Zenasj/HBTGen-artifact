import torch

torch.LongTensor([1], device = 'cuda')
# RuntimeError: legacy constructor for device type: cpu was passed device type: cuda, but device type must be: cpu