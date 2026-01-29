# torch.rand(B, C, H, W, dtype=torch.float32)  # input shape (1,5,3,10)
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(10, 20),
            nn.BatchNorm2d(5),
            nn.ReLU()
        )
        self.eval()  # Set to eval mode as in the original example

    def forward(self, x):
        return self.model(x)

def my_model_function():
    return MyModel()

def GetInput():
    return torch.rand(1, 5, 3, 10, dtype=torch.float32)

