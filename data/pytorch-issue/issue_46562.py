# torch.rand(B, 1, H, W, dtype=torch.float32)  # Inferred input shape from Conv2d(1,1,1) in Sequential

import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(1, 1, 1)
        )
        
    def forward(self, x):
        return self.model(x)

def my_model_function():
    # Returns the model instance in eval mode as in the original issue
    model = MyModel()
    model.eval()
    return model

def GetInput():
    # Returns a random tensor matching the input expected by MyModel
    # (batch, channels=1, arbitrary H/W dimensions)
    return torch.rand(1, 1, 32, 32, dtype=torch.float32)

