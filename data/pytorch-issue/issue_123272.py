# torch.rand(B, 2, dtype=torch.float32)
import torch
import torch.nn as nn

class TPModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.ffn1 = nn.Linear(8, 4, bias=False)  # fs_dim=8, tp_dim=4 from original code

    def forward(self, x):
        return self.ffn1(x)

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.net1 = nn.Linear(2, 8, bias=False)  # input_dim=2
        self.net2 = nn.Linear(8, 8, bias=False)
        self.ffn = TPModel()  # TPModel as submodule

    def forward(self, x):
        x = self.net1(x)
        x = self.net2(x)
        return self.ffn(x)

def my_model_function():
    return MyModel()

def GetInput():
    return torch.rand(4, 2, dtype=torch.float32)  # Matches original input dimensions

