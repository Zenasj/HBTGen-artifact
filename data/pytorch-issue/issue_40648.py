# torch.rand(B, 4, dtype=torch.float32, device='cuda:0')
import torch
from torch import nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        # Create a sparse COO tensor with indices and values on the same device
        indices = torch.tensor([[0, 1, 2], [0, 1, 3]], device='cuda:0', dtype=torch.long)
        values = torch.rand(3, device='cuda:0')
        shape = (4, 4)
        self.sparse_param = torch.sparse_coo_tensor(indices, values, shape)
        self.sparse_param = nn.Parameter(self.sparse_param)

    def forward(self, x):
        # Multiply input with sparse parameter (assuming x is (B,4))
        # Transpose to (4, B) to match sparse matrix multiplication
        return torch.sparse.mm(self.sparse_param, x.t()).t()

def my_model_function():
    return MyModel()

def GetInput():
    # Generate input tensor matching the model's expectation
    B = 2  # Arbitrary batch size
    return torch.rand(B, 4, dtype=torch.float32, device='cuda:0')

