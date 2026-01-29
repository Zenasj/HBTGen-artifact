# torch.rand(4, 4, dtype=torch.float32)  # Input shape for the model

import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.seq = nn.Sequential(
            nn.Linear(4, 4),
            nn.Linear(4, 4),
            nn.Linear(4, 4),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq_last = self.seq[1:]  # Slicing the sequential module
        return seq_last(x)

def my_model_function():
    # Return an instance of MyModel
    return MyModel()

def GetInput():
    # Return a random tensor input that matches the input expected by MyModel
    return torch.randn(4, 4, dtype=torch.float32)

