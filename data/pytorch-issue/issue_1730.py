# torch.rand(1, 1, 91, 91, dtype=torch.float)
import torch
from torch import nn

class InplaceAdd(nn.Module):
    def forward(self, source):
        target = torch.zeros_like(source)
        target.add_(source)
        return target

class NonInplaceAdd(nn.Module):
    def forward(self, source):
        target = torch.zeros_like(source)
        return target + source  # Non-inplace addition

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.inplace_add = InplaceAdd()
        self.noninplace_add = NonInplaceAdd()
    
    def forward(self, source):
        inplace_result = self.inplace_add(source)
        noninplace_result = self.noninplace_add(source)
        # Return comparison result as a float tensor (1.0 for match, 0.0 otherwise)
        return torch.tensor(
            torch.allclose(inplace_result, noninplace_result, atol=1e-5),
            dtype=torch.float
        )

def my_model_function():
    return MyModel()

def GetInput():
    # Returns a 4D tensor matching the input expected by MyModel
    return torch.ones(1, 1, 91, 91, dtype=torch.float)

