# torch.rand(B, 10, dtype=torch.float32)
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Sequential(
                nn.Linear(10, 20),
                nn.ReLU(),
                nn.Linear(20, 1)
            ),
            nn.Sequential(
                nn.Linear(1, 20),
                nn.ReLU(),
                nn.Linear(20, 1)
            )
        )

    def forward(self, x):
        return self.net(x)

def my_model_function():
    return MyModel()

def GetInput():
    return torch.rand(1, 10, dtype=torch.float32)

