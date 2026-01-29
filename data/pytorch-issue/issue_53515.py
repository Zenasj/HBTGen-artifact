import torch
from torch import nn

# torch.rand(B, 1, dtype=torch.float32, device='cuda')
class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.model1 = Model1()
        self.model2 = Model2()

    def forward(self, x):
        out1 = self.model1(x)
        out2 = self.model2(x)
        return out1, out2  # Return both outputs for comparison

class Model1(nn.Module):  # Example1's forward logic
    def __init__(self):
        super().__init__()
        self.model = nn.Linear(1, 1, bias=False).to("cuda")

    def forward(self, x):
        param = next(self.parameters())
        param.requires_grad = True
        x = self.model(x).mean()
        param.requires_grad = False
        return x

class Model2(nn.Module):  # Example2's forward logic
    def __init__(self):
        super().__init__()
        self.model = nn.Linear(1, 1, bias=False).to("cuda")

    def forward(self, x):
        param = next(self.parameters())
        param.requires_grad = False
        x = self.model(x).mean()
        param.requires_grad = True
        return x

def my_model_function():
    return MyModel()

def GetInput():
    return torch.rand(1, device="cuda", dtype=torch.float32)

