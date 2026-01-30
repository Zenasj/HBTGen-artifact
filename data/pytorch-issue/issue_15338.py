import torch
a = torch.tensor([float('nan')])
print(a.clamp(0, 1)) # prints nan
print(a.cuda().clamp(0, 1)) # prints 1