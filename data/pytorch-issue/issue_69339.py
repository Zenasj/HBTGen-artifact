import torch.nn as nn

import torch
from torch.nn import CrossEntropyLoss
CrossEntropyLoss(weight=torch.tensor([.2, .3]), label_smoothing=0.1)(torch.tensor([[1, 2], [3, .4]]), torch.tensor([-100, 1]))

CrossEntropyLoss(label_smoothing=0.1)(torch.tensor([[1, 2], [3, .4]]), torch.tensor([-100, 1]))

CrossEntropyLoss(weight=torch.tensor([.2, .3]))(torch.tensor([[1, 2], [3, .4]]), torch.tensor([-100, 1]))

CrossEntropyLoss(weight=torch.tensor([.2, .3]), label_smoothing=0.1)(torch.tensor([[1, 2], [3, .4]]), torch.tensor([0, 1]))

import torch
output = torch.nn.CrossEntropyLoss(ignore_index=1)(torch.tensor([[1, 2], [3, .4]]), torch.tensor([0, 1]))
print(output)