# torch.rand(B, 100, dtype=torch.float32)
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(100, 100, bias=False),
            nn.Sequential(
                nn.Linear(100, 100, bias=False),
                nn.Sequential(
                    nn.Linear(100, 100, bias=False),
                    nn.Sequential(
                        nn.Linear(100, 100, bias=False)
                    )
                )
            )
        )
    
    def forward(self, x):
        return self.model(x)

def my_model_function():
    return MyModel()

def GetInput():
    return torch.rand(1, 100, dtype=torch.float32).cuda()

