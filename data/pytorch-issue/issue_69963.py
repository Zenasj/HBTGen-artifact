# torch.rand(1, 1024*1024, 3, dtype=torch.float32)
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc_in = nn.Linear(3, 128)
        self.layers = nn.ModuleList()
        for _ in range(10):
            self.layers.append(nn.Linear(128, 128))
            self.layers.append(nn.LayerNorm(128))
        self.fc_out = nn.Linear(128, 3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc_in(x)
        for layer in self.layers:
            x = layer(x)
        x = self.fc_out(x)
        return x

def my_model_function():
    return MyModel()

def GetInput():
    return torch.rand(1, 1024 * 1024, 3, dtype=torch.float32)

