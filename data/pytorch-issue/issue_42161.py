import torch
a = torch.tensor([3., 4.]).to(torch.device('cuda:0'))
b: torch.Tensor = torch.tensor(2.)
b * a

import torch
a = torch.tensor([3., 4.]).to(torch.device('cuda:1'))
b: torch.Tensor = torch.tensor(2.)
b * a