# torch.rand(B, C, H, W, dtype=torch.float32)
import torch
from torch import nn

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.fc = nn.Linear(1, 2)
        self.pool = nn.AdaptiveMaxPool3d(output_size=[0, 0, 1], return_indices=True)

    def forward(self, x):
        x = self.fc(x)
        print('After fc', x.size())
        x = self.pool(x)
        return x

def my_model_function():
    return MyModel()

def GetInput():
    return torch.rand([2, 2, 1, 1])

