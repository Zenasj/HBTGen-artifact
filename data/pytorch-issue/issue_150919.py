# torch.rand(2, 3, dtype=torch.complex64)  # Input shape inferred as two vectors of length 3
import torch
from torch import nn

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.reference = ReferenceModule()
        self.torch_module = TorchModule()

    def forward(self, x):
        a, b = x[0], x[1]  # Split into two vectors
        a_batch = a.unsqueeze(0)  # Add batch dimension for module processing
        b_batch = b.unsqueeze(0)
        
        # Compute precise result using double precision
        a_d = a_batch.double()
        b_d = b_batch.double()
        precise = torch.linalg.vecdot(a_d, b_d, dim=-1)
        
        # Compute reference and torch outputs
        ref_out = self.reference(a_batch, b_batch)
        torch_out = self.torch_module(a_batch, b_batch)
        
        # Calculate distances in double precision
        ref_dist = torch.abs(ref_out.double() - precise)
        torch_dist = torch.abs(torch_out.double() - precise)
        
        # Return whether Torch result is closer to precise value
        return torch_dist < ref_dist

class ReferenceModule(nn.Module):
    def forward(self, a, b):
        return torch.sum(a * torch.conj(b), dim=-1)

class TorchModule(nn.Module):
    def forward(self, a, b):
        return torch.linalg.vecdot(a, b, dim=-1)

def my_model_function():
    return MyModel()

def GetInput():
    return torch.rand(2, 3, dtype=torch.complex64)  # Two vectors of length 3 in complex64

