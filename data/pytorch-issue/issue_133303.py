# torch.rand(2, dtype=torch.float32)
import torch
import torch.nn as nn

def hook_fn(module, input, output):
    return None

class SubM(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(2, 2)
        self.register_full_backward_hook(hook_fn)
    
    def forward(self, x):
        return self.fc(x)
    
class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(2, 2)
        self.fc2 = nn.Linear(2, 2)
        self.predictions = SubM()

    def forward(self, x):
        x = self.predictions(x)
        x = self.fc1(x)
        return self.fc2(x)

def my_model_function():
    return MyModel()

def GetInput():
    return torch.rand(2, requires_grad=True)

