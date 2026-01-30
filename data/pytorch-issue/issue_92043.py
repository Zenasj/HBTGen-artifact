import torch
a = torch.tensor(0.0 + 0.0j)
b = torch.tensor(1e-36 + 0.0j)
print(a / b)  # it's (nan + nanj)