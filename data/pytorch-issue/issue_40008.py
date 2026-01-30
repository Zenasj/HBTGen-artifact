import torch
import torch.nn as nn

# Disable automatic differentiation
torch.autograd.set_grad_enabled(False)
# Even enter a no gradient context
with torch.no_grad():
  # In your model, setup a parameter:
  self.l1w = nn.Parameter(torch.randn(num_inputs, 20))
# l1w requires a gradient
assert self.l1w.requires_grad == True