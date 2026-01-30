import torch.nn.functional as F

import torch.distributions as distrib
import torch.distributions.transforms as transform

class PlanarFlow(transform.Transform):
    def __init__(self, weight, scale, bias):
        super(PlanarFlow, self).__init__()
        self.bijective = False

        # Add these 2 lines
        self.domain = torch.distributions.constraints.Constraint()
        self.codomain = torch.distributions.constraints.Constraint()
        
        self.weight = weight
        self.scale = scale
        self.bias = bias

    def _call(self, z):
        f_z = F.linear(z, self.weight, self.bias)
        return z + self.scale * torch.tanh(f_z)

    def log_abs_det_jacobian(self, z):
        f_z = F.linear(z, self.weight, self.bias)
        psi = (1 - torch.tanh(f_z) ** 2) * self.weight
        det_grad = 1 + torch.mm(psi, self.scale.t())
        return torch.log(det_grad.abs() + 1e-7)