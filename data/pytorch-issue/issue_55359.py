import torch

t = torch.tensor(complex(2, 3))
print(torch.sigmoid(t))
print(torch.sigmoid(t.cuda()))