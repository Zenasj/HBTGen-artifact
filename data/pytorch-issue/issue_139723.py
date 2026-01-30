import torch.nn as nn

import torch

loss = torch.nn.MSELoss()

a = torch.tensor([0]).to("mps")
b = torch.tensor([0]).to("mps")

loss(a, b)