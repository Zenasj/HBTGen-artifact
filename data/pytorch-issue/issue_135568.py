# torch.rand(B, 2, dtype=torch.float32)
import torch
import torch.nn as nn

class Level2(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(3, 2)

    def forward(self, x):
        return self.linear(x)
    
class Level3(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(2, 2)
        self.non_linear = nn.ReLU()

    def forward(self, x):
        x = self.linear(x)
        return self.non_linear(x)

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(2, 3)
        self.linear2 = Level2()
        self.linear3 = Level3()

    def forward(self, x):
        x = self.linear1(x)
        x = self.linear2(x)
        return self.linear3(x)

def my_model_function():
    return MyModel()

def GetInput():
    # Generate random batch size between 2-9 (as per original test loop)
    B = torch.randint(2, 10, (1,)).item()
    return torch.rand(B, 2, dtype=torch.float32)

