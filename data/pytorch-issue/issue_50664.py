# torch.rand(B, 1000, dtype=torch.float32)
import torch
import torch.nn as nn
import torch.nn.functional as F

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.layers = nn.ModuleList()
        for _ in range(100):
            layer = nn.Linear(1000, 1000, bias=False)
            # Use SELU gain value of 0.75 as per PR implementation
            nn.init.xavier_normal_(layer.weight, gain=nn.init.calculate_gain("selu"))
            self.layers.append(layer)
    
    def forward(self, x):
        for layer in self.layers:
            x = F.selu(layer(x))
        return x

def my_model_function():
    return MyModel()

def GetInput():
    # Matches input shape (batch_size, 1000 features)
    return torch.rand(2, 1000, dtype=torch.float32)

