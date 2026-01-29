# torch.rand(B, 3, dtype=torch.float32)
import copy
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Linear(3, 1024)
        self.ema_model = copy.deepcopy(self.model)  # Fused models as submodules
        
    def forward(self, x):
        # Return outputs of both models for comparison
        return self.model(x), self.ema_model(x)

def my_model_function():
    return MyModel()

def GetInput():
    # Random input matching the Linear layer's input shape (batch, in_features=3)
    return torch.rand(2, 3, dtype=torch.float32)

