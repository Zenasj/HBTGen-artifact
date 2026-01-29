# torch.rand(1, 5, 4, dtype=torch.double, device='cuda').to_sparse()
import torch
from torch import nn

class MyModel(nn.Module):
    def forward(self, S):
        S = S.coalesce()  # Explicitly coalesce as in the test case
        S.requires_grad_(True)  # Trigger the reported memory leak condition
        return torch.sparse.sum(S)

def my_model_function():
    return MyModel()

def GetInput():
    device = 'cuda'
    indices = torch.tensor([[0], [4]], dtype=torch.long, device=device)
    values = torch.tensor([[[-0.4567, -1.8797, 0.0380, 1.4316]]], 
                         dtype=torch.double, device=device)
    # Create sparse COO tensor with inferred shape (1,5,4) from indices and values
    return torch.sparse_coo_tensor(indices, values, dtype=torch.double, device=device)

