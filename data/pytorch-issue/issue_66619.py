# torch.rand(B, 5, 1, 1, dtype=torch.float32)
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        # Values tensor for heaviside (fixed at 0.5 as in the example)
        self.register_buffer('values', torch.tensor(0.5))

    def forward(self, x):
        # Apply heaviside function, which causes backward error in PyTorch 1.9.0+
        return torch.heaviside(x, self.values)

def my_model_function():
    return MyModel()

def GetInput():
    B = 1  # Batch size (can be adjusted, minimal for reproducibility)
    return torch.rand(B, 5, 1, 1, dtype=torch.float32, requires_grad=True)

