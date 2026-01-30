import torch
import torch.nn as nn

x = torch.randn(4,2,requires_grad=True)
fc = nn.Linear(2, 1)

#x.detach_().requires_grad_()

def _save_output(module, grad_input, grad_output) -> None:
  print("grad_output[0]: ",grad_output[0])

fc.register_full_backward_hook(_save_output)

with torch.no_grad():
  y = fc(x)

z = x.pow(2).sum(dim=-1)

#loss is defined as the sum of 2 terms
#first term, y has no gradient to it (purely scales)
#second term, z has a gradient. (in short, y scales z)
loss = torch.sum(y+z)

loss.backward()