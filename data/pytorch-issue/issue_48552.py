import torch
x = torch.tensor(1, dtype=torch.cfloat)
x.requires_grad = True
y = torch.sigmoid(x)
y.backward()