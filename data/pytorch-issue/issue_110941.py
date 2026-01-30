import torch

I,J,K = 5,3,7

x1 = torch.zeros(I,J,1).bool()
x2 = torch.zeros(I,1,K).bool()
torch.logical_and(x1, x2)

import torch

I,J,K = 5,3,7

x1 = torch.zeros(I,J,1).bool()
x2 = torch.zeros(I,1,K).bool()
x1*x2