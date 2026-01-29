# torch.rand(10, 10, dtype=torch.float32)
import torch
import warnings

class MyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = torch.nn.Linear(10, 10)

    def forward(self, x):
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            return self.layer(x)

def my_model_function():
    return MyModel()

def GetInput():
    return torch.randn(10, 10)

