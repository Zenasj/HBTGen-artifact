import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

torch.manual_seed(1)
dev = 'cpu'

k = torch.ones(3, 3, 9, 9).to(dev)

x = torch.rand(1, 3, 32, 32).to(dev)
x = F.pad(x, (2, 2, 2, 2), mode='circular')

y = F.conv2d(x, k)
plt.imshow(y[0, 0].detach().cpu())
plt.show()

torch.manual_seed(1)
dev = 'mps'

k = torch.ones(3, 3, 9, 9).to(dev)

x = torch.rand(1, 3, 32, 32).to(dev)
x = F.pad(x, (2, 2, 2, 2), mode='circular')

y = F.conv2d(x, k)
plt.imshow(y[0, 0].detach().cpu())
plt.show()