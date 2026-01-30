import torch

torch.save(torch.zeros([309237982, 2, 5], dtype=torch.int8), './sample.pt')
torch.load('./sample.pt')