import torch.nn as nn
import numpy as np

import torch
input = torch.from_numpy(np.array([0.2736]))
input.requires_grad = True
out = torch.nn.Softplus(beta=3, threshold=1)(input)
out = out.sum()
out.backward()
print(input.grad.numpy().tolist())