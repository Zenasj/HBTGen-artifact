# torch.rand(5000000, 256, dtype=torch.bfloat16, device="cuda")
import torch
from torch import nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(256, 16)

    def forward(self, x):
        return self.fc(x)

def my_model_function():
    model = MyModel()
    model.eval()
    model.to(dtype=torch.bfloat16, device="cuda")
    return model

def GetInput():
    return torch.rand(5000000, 256, dtype=torch.bfloat16, device="cuda")

