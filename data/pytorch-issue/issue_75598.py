# torch.rand(1, 2048, dtype=torch.int64)  # Inferred input shape from the issue

import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        # Assuming the model has some layers before and after the amax operation
        # For simplicity, we will use an identity layer to represent these
        self.identity = nn.Identity()

    def forward(self, tokens):
        sim = self.identity(tokens)
        # Perform the amax operation
        max_sim = sim.amax(dim=-1, keepdim=True).detach()
        sim = sim - max_sim
        return sim

def my_model_function():
    # Return an instance of MyModel, include any required initialization or weights
    return MyModel()

def GetInput():
    # Return a random tensor input that matches the input expected by MyModel
    return torch.randint(0, 20000, (1, 2048), dtype=torch.int64)

