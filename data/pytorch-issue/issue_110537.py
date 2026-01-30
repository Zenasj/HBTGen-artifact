import torch
import torch.nn as nn

class MyModule(nn.Module):
    def __init__(self, x, y):
        super().__init__()
        self.register_parameter("x", nn.Parameter(x))
        self.register_buffer("y", y)

x = torch.rand(3, dtype=torch.complex64)
y = torch.rand(3, dtype=torch.complex64)
model = MyModule(x, y).to(device="mps")
model.x * model.y

import torch
x = torch.ones(3, dtype=torch.complex64, device='mps')
# TypeError: Trying to convert ComplexFloat to the MPS backend but it does not have support for that dtype.

x = torch.ones(3, dtype=torch.complex64)
x = x.to(device='mps')