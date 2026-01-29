# torch.rand(1, 16, dtype=torch.float32).to_sparse_csr()  # Input shape and dtype
import torch
from torch import nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(16, 8)
        self.fc2 = nn.Linear(8, 4)

    def forward(self, x):
        # Convert sparse CSR to dense before applying dense layers
        x = x.to_dense()
        x = self.fc1(x)
        x = self.fc2(x)
        return x

def my_model_function():
    return MyModel()

def GetInput():
    # Returns a random sparse CSR tensor matching the expected input shape
    return torch.rand(1, 16, dtype=torch.float32).to_sparse_csr()

