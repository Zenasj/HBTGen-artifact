# torch.rand(1, 15, dtype=torch.float32) ← Add a comment line at the top with the inferred input shape
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.linear = nn.Linear(15, 30)

    def forward(self, x: torch.Tensor):
        x *= x
        y = self.linear(x)
        y += 3
        y -= 1
        z = torch.mean(y)
        return z

def my_model_function():
    # Return an instance of MyModel, include any required initialization or weights
    return MyModel()

def GetInput():
    # Return a random tensor input that matches the input expected by MyModel
    return torch.rand(1, 15, dtype=torch.float32)

