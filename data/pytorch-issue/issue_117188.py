# torch.rand(64, 64, dtype=torch.float32)
import torch
from torch import nn

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
    
    def forward(self, x):
        return x.sum()

def my_model_function():
    return MyModel()

def GetInput():
    # Returns a sparse CSR tensor to match the issue's scenario
    dense = torch.rand(64, 64, dtype=torch.float32)
    return dense.to_sparse_csr()

