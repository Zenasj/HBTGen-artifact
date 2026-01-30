import torch.nn as nn

with torch.context(device='cuda:2', dtype=torch.double):
   x = torch.ones((5, 5))

# model.py

import torch

class Model(torch.nn.Module):
  ...

# main.py

torch_ctx = torch.get_context(sys.argv[1])

from model import Model

m = Model()  # <- not using `torch_ctx`

torch_ctx = torch.get_context(sys.argv[1])

import model
model = torch_ctx.wrap(model)

m = model.Model()