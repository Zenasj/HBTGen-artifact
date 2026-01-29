# torch.rand(B, 2, dtype=torch.float32, device='cuda')  # Input shape: (batch_size, 2)

import torch
import torch.nn as nn
import math

class MyModel(nn.Module):
    def __init__(self, mean, scale_tril):
        super(MyModel, self).__init__()
        self.mean = mean
        self.scale_tril = scale_tril
        self.dist = torch.distributions.MultivariateNormal(self.mean, scale_tril=self.scale_tril)

    def forward(self, inputs):
        p = self.mean.size(0)
        diff = inputs - self.mean
        
        batch_shape = diff.shape[:-1]
        
        scale_shape = self.scale_tril.size()
        
        _scale_tril = self.scale_tril.expand(batch_shape + scale_shape)
        z = torch.linalg.solve_triangular(_scale_tril,
                                          diff.unsqueeze(-1), 
                                          upper=False).squeeze()
        
        out = -0.5 * p * torch.tensor(2 * math.pi).log() - _scale_tril.logdet() - 0.5 * (z ** 2).sum(dim=-1)
        return out.squeeze()

def my_model_function():
    device = torch.device("cuda")
    dtype = torch.float32
    mean = torch.tensor([0.0, 0.0], dtype=dtype, device=device)
    scale_tril = torch.diag_embed(torch.tensor([1.0, 1.0], dtype=dtype, device=device))
    return MyModel(mean, scale_tril)

def GetInput(batch_size=524281):
    device = torch.device("cuda")
    dtype = torch.float32
    return torch.randn([batch_size, 2], dtype=dtype, device=device)

