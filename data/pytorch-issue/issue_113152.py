# torch.rand(B, 100, dtype=torch.float32)
import torch
from torch import nn
from collections import OrderedDict

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        # Share the same modules between both Sequential instances to ensure identical behavior
        self.layer1 = nn.Linear(100, 50)
        self.relu = nn.ReLU()
        self.layer2 = nn.Linear(50, 10)
        
        # Model without explicit names (standard Sequential)
        self.model1 = nn.Sequential(
            self.layer1,
            self.relu,
            self.layer2,
        )
        
        # Model with explicit names (using OrderedDict)
        self.model2 = nn.Sequential(
            OrderedDict([
                ('layer1', self.layer1),
                ('relu', self.relu),
                ('layer2', self.layer2),
            ])
        )
    
    def forward(self, x):
        out1 = self.model1(x)
        out2 = self.model2(x)
        # Return boolean tensor indicating if outputs are numerically identical
        return torch.tensor(torch.allclose(out1, out2), dtype=torch.bool).unsqueeze(0)

def my_model_function():
    return MyModel()

def GetInput():
    B = 2  # Arbitrary batch size
    return torch.rand(B, 100, dtype=torch.float32)

