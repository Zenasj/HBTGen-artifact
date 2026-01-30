import torch.nn as nn

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

x = torch.ones(3, 2, 1, requires_grad=True)

model = nn.Sequential(
    nn.InstanceNorm1d(2),
    nn.ReLU(inplace=True),
)
y = checkpoint(model, x)

y.norm().backward()

print(x.grad)
# Output: None
# Expected: tensor([[[0.],[0.]], ...])

# No in-place operation
model = nn.Sequential(
    nn.InstanceNorm1d(2),
    nn.ReLU(inplace=False),
)
y = checkpoint(model, x)

# No checkpoint
model = nn.Sequential(
    nn.InstanceNorm1d(2),
    nn.ReLU(inplace=True),
)
y = model(x)

# Not at the end of checkpoint
model = nn.Sequential(
    nn.InstanceNorm1d(2),
    nn.ReLU(inplace=True),
    nn.Conv1d(2, 2, 1),
)
y = checkpoint(model, x)

# Other norm
model = nn.Sequential(
    nn.BatchNorm1d(2),
    nn.ReLU(inplace=True),
)
y = checkpoint(model, x)

def run_fn(input):
    out = input + 1             # add a grad_fn
    out = out.view(out.size())  # make a view
    out = out.relu_()           # apply in-place op
    return out

input_var = torch.rand(1, requires_grad=True)
out = checkpoint(run_fn, input_var)
out.backward()

assert input_var.grad is not None  # FAILS!
assert out.grad_fn.__class__.__name__ == 'CheckpointFunctionBackward'  # FAILS!