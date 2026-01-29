# (torch.rand(3), torch.rand(1))  # input shape for x and t respectively
import torch
from torch.func import jacrev
import torch._dynamo  # Required for allow_in_graph decorator

class MyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.coeff = 3.0  # Matches the example function's multiplier

    def f(self, x, t):
        return self.coeff * t * x  # Example function implementation

    @torch._dynamo.allow_in_graph
    def forward(self, inputs):
        x, t = inputs
        # Compute Jacobian w.r. to x (argnums=0) and trace it
        jac = jacrev(self.f, argnums=0)(x, t)
        return torch.trace(jac)

def my_model_function():
    return MyModel()

def GetInput():
    return (torch.rand(3), torch.rand(1))  # Matches example input shapes

