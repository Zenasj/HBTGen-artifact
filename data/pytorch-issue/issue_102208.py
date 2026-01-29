# torch.rand(B, 2, dtype=torch.float32)  # Input shape: batch_size x 2 (start and end values)
import torch
from torch import nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.steps = 10  # Fixed steps as in the original example's partial(torch.linspace, steps=10)

    def forward(self, inputs):
        start, end = inputs.unbind(dim=1)  # Split into (B,) tensors for start and end
        steps = self.steps
        # Compute custom linspace using arange to avoid .item() usage
        step_size = (end - start) / (steps - 1)
        range_tensor = torch.arange(steps, device=inputs.device, dtype=inputs.dtype)
        return start[:, None] + step_size[:, None] * range_tensor[None, :]

def my_model_function():
    return MyModel()

def GetInput():
    B = 3  # Example batch size matching original issue's start/stop tensors
    return torch.rand(B, 2, dtype=torch.float32)  # (batch_size, 2) tensor for start/end values

