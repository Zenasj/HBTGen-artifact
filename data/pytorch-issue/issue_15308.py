import torch

device = torch.device('cuda:1')
a = torch.tensor(5, device=device)
torch.save({'a':a}, 'test.th')