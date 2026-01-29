# torch.rand(B, 2, dtype=torch.float32)  # Input shape: batch_size x input_dim (2)
import torch
import torch.nn as nn

fs_dim = 8
tp_dim = 4
input_dim = 2

class TPModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.ffn1 = nn.Linear(fs_dim, tp_dim, bias=False)
    
    def forward(self, x):
        return self.ffn1(x)
    
    def fsdp_wrap_fn(self, fsdp_mesh):
        return {'device_mesh': fsdp_mesh}

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.net1 = nn.Linear(input_dim, fs_dim, bias=False)
        self.net2 = nn.Linear(fs_dim, fs_dim, bias=False)
        self.ffn = TPModel()  # TPModel is encapsulated as a submodule
    
    def forward(self, x):
        return self.ffn(self.net2(self.net1(x)))

def my_model_function():
    return MyModel()

def GetInput():
    return torch.rand(4, input_dim, dtype=torch.float32)  # Matches input_dim=2 and batch_size=4 from original code

