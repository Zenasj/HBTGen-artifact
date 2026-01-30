bash
import torch
import torch.nn as nn
t = torch.rand(2,5)
nn.Linear(2,2)(t).shape

bash
t = torch.rand(2,5, device='cuda')
nn.Linear(2,2, device='cuda')(t).shape