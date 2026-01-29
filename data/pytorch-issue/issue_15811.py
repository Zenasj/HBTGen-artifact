# torch.rand(B, C, dtype=torch.float32)
import torch
import torch.nn as nn

class BugModule(nn.Module):
    def __init__(self, num_mod):
        super().__init__()
        self.modlist = nn.ModuleList([nn.Linear(1000, 50) for _ in range(num_mod)])

    def forward(self, x):
        return self.modlist[0](x)

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.small = BugModule(2)  # 2 modules (baseline)
        self.large = BugModule(200)  # 200 modules (problem case)

    def forward(self, x):
        # Return outputs of both submodels to compare their execution
        out_small = self.small(x)
        out_large = self.large(x)
        return out_small, out_large

def my_model_function():
    return MyModel()

def GetInput():
    return torch.rand(50, 1000, dtype=torch.float32)

