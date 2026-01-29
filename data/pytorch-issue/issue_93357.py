# torch.rand(1)  # Dummy input not used by the model
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.p1 = torch.nn.parameter.Parameter(
            torch.tensor([4, 4, 5, 5, 4, 4, 4, 7, 4]), requires_grad=False
        )

    def forward(self, x):  # Accepts dummy input to comply with GetInput()
        v0 = torch.cat([self.p1, self.p1], dim=0)  # Concatenates self.p1 twice
        v5 = v0.expand(1, 1, 1, 18)  # Expands to (1,1,1,18) shape
        return v5

def my_model_function():
    return MyModel()

def GetInput():
    return torch.rand(1)  # Dummy input tensor (not used by the model)

