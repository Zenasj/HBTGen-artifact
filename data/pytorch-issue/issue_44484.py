import torch

from torch.utils._benchmark import Timer
m = Timer(
    "x.backward()", 
    globals={"x": torch.ones((1,)) + torch.ones((1,), requires_grad=True)}
).blocked_autorange()
print(m)