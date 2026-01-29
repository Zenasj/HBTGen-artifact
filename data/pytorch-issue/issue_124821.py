# torch.rand(22, 51, dtype=torch.int64) ← Add a comment line at the top with the inferred input shape
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.v2_0 = torch.nn.Parameter(torch.empty([1, 22, 51], dtype=torch.int64), requires_grad=False)

    def forward(self, _args):
        v2_0 = self.v2_0
        getitem = _args
        max_1 = getitem.max(0)
        getattr_1 = max_1.values
        max_2 = torch.max(getitem, v2_0)
        return (getattr_1, max_2)

def my_model_function():
    # Return an instance of MyModel, include any required initialization or weights
    return MyModel()

def GetInput():
    # Return a random tensor input that matches the input expected by MyModel
    return torch.randint(0, 100, (22, 51), dtype=torch.int64)

