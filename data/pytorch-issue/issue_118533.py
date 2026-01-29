# torch.rand(B, C, H, W, dtype=torch.float32)
import torch
from torch import nn, Tensor

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(224*224*3, 10)  # Example layer matching input shape

    def forward(self, x: Tensor) -> Tensor:
        is_batched = x.dim() == 4
        if not is_batched:
            x = x.unsqueeze(0)  # Ensure batch dimension
        x = x.view(x.size(0), -1)
        return self.linear(x)

def my_model_function():
    return MyModel()

def GetInput():
    return torch.rand(1, 3, 224, 224, dtype=torch.float32)

