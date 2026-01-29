# torch.rand(1, dtype=torch.float32)  # Dummy input tensor
import torch
from torch import nn

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.param = nn.Parameter(torch.sparse_coo_tensor(size=(3,), dtype=torch.float32))
    
    def forward(self, x):
        # Convert sparse parameter to dense for compatibility with loss computation
        return self.param.to_dense()

def my_model_function():
    return MyModel()

def GetInput():
    return torch.rand(1)  # Dummy input to satisfy model's __call__ signature

