import torch
import torch.nn as nn

torch.manual_seed(1)

x = torch.rand(10, 8, device='mps')
x = x / x.sum(dim=1, keepdim=True)
x = log_softmax(x, dim=-1)

y = torch.rand(10, 8, device='mps')
y = y / y.sum(dim=1, keepdim=True)

criterion = nn.KLDivLoss(reduction="sum")
print(criterion(x, y), criterion(x.to('cpu'), y.to('cpu')))

# mask out random entries of y
mask = torch.rand(10, 8, device='mps') < 0.5
y = y * mask
print(criterion(x, y), criterion(x.to('cpu'), y.to('cpu')))