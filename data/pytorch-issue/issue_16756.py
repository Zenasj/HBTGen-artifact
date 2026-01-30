import torch
import torch.nn as nn
import torch.nn.functional as F

class PlanarFlow(tr.Transform, nn.Module):

    def __init__(self, dim):
        tr.Transform.__init__(self)
        nn.Module.__init__(self)
        self.weight = nn.Parameter(torch.Tensor(1, dim))
        self.bias = nn.Parameter(torch.Tensor(1))
        self.scale = nn.Parameter(torch.Tensor(1, dim))

    def _call(self, z):
        f_z = F.linear(z, self.weight, self.bias)
        return z + self.scale * F.tanh(f_z)

    def log_abs_det_jacobian(self, z):
        f_z = F.linear(z, self.weight, self.bias)
        psi = (1 - F.tanh(f_z) ** 2) * self.weight
        det_grad = 1 + torch.mm(psi, self.scale.t())
        return torch.log(det_grad.abs() + 1e-9)

# Main class for normalizing flow
class NormalizingFlow(nn.Module):

    def __init__(self, dim, flow_length, density):
        super().__init__()
        biject = []
        for f in range(flow_length):
            biject.append(PlanarFlow(dim))
        self.bijectors = ComposeTransform(biject)
        self.modules = nn.ModuleList(biject)
        self.base_density = density
        self.final_density = TransformedDistribution(density, self.bijectors)
        self.log_det = []

    def forward(self, z):
        self.log_det = []
        # Applies series of flows
        for bijector in self.bijector:
            self.log_det.append(bijector.log_abs_det_jacobian(z))
            z = bijector(z)
        return z, self.log_det

class Flow(tr.Transform, nn.Module):
    
    def __init__(self):
        tr.Transform.__init__(self)
        nn.Module.__init__(self)
        
    # Hacky hash bypass
    def __hash__(self):
        return nn.Module.__hash__(self)

class PlanarFlow(Flow):

    def __init__(self, dim):
        super(PlanarFlow, self).__init__()
        self.weight = nn.Parameter(torch.Tensor(1, dim))
        self.bias = nn.Parameter(torch.Tensor(1))
        self.scale = nn.Parameter(torch.Tensor(1, dim))

    def _call(self, z):
        f_z = F.linear(z, self.weight, self.bias)
        return z + self.scale * F.tanh(f_z)

    def log_abs_det_jacobian(self, z):
        f_z = F.linear(z, self.weight, self.bias)
        psi = (1 - F.tanh(f_z) ** 2) * self.weight
        det_grad = 1 + torch.mm(psi, self.scale.t())
        return torch.log(det_grad.abs() + 1e-9)