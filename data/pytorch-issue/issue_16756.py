# torch.rand(B, 2, dtype=torch.float)  # Input shape: (batch, dim=2)
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Transform, Independent, Normal
from torch.distributions.transforms import ComposeTransform

class Flow(Transform, nn.Module):
    def __init__(self):
        Transform.__init__(self)
        nn.Module.__init__(self)
        
    def __hash__(self):
        return nn.Module.__hash__(self)

class PlanarFlow(Flow):
    def __init__(self, dim):
        super().__init__()
        self.weight = nn.Parameter(torch.Tensor(1, dim))
        self.bias = nn.Parameter(torch.Tensor(1))
        self.scale = nn.Parameter(torch.Tensor(1, dim))
        self.reset_parameters()
    
    def reset_parameters(self):
        nn.init.normal_(self.weight, 0, 0.01)
        nn.init.constant_(self.bias, 0.0)
        nn.init.normal_(self.scale, 0, 0.01)
    
    def _call(self, z):
        f_z = F.linear(z, self.weight, self.bias)
        return z + self.scale * torch.tanh(f_z)
    
    def log_abs_det_jacobian(self, z):
        f_z = F.linear(z, self.weight, self.bias)
        psi = (1 - torch.tanh(f_z) ** 2) * self.weight
        det_grad = 1 + torch.mm(psi, self.scale.t())
        return torch.log(torch.abs(det_grad) + 1e-9)

class MyModel(nn.Module):
    def __init__(self, dim=2, flow_length=3):
        super().__init__()
        self.dim = dim
        bijectors = []
        for _ in range(flow_length):
            bijectors.append(PlanarFlow(dim))
        self.bijector_sequence = ComposeTransform(bijectors)
        self.bijectors = nn.ModuleList(bijectors)  # Track parameters
        self.base_dist = Independent(Normal(torch.zeros(dim), torch.ones(dim)), 1)
    
    def forward(self, z):
        return self.bijector_sequence(z)

def my_model_function():
    return MyModel()

def GetInput():
    B = 4  # Arbitrary batch size
    return torch.rand(B, 2, dtype=torch.float)

