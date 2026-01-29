# torch.rand(npos, 2, dtype=torch.float32, device="cuda:0")  # Add a comment line at the top with the inferred input shape

import torch
from torch import nn

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.logits = nn.Parameter(torch.zeros((1000, 2), device="cuda:0", requires_grad=True))

    def forward(self, targets):
        loss = nn.functional.cross_entropy(self.logits, targets)
        return loss

def my_model_function():
    # Return an instance of MyModel, include any required initialization or weights
    return MyModel()

def GetInput():
    # Return a random tensor input that matches the input expected by MyModel
    npos = 1000
    targets = torch.zeros((), dtype=torch.long, device="cuda:0").expand(npos)
    return targets

