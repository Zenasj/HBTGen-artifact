# torch.rand(2, 12, 2, dtype=torch.float32)
import torch
import torch.nn as nn
import torch.nn.functional as F

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        # Target tensor from the issue's repro steps
        self.register_buffer('target', torch.tensor([
            [1, 0, 0, 0, 0, -1, -1, -1, -1, -1, -1, -1],
            [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0]
        ], dtype=torch.long))

    def forward(self, input):
        input_flat = input.view(-1, input.size(-1))
        target_flat = self.target.view(-1)
        # Reproduce the problematic loss computation
        return F.nll_loss(input_flat, target_flat, ignore_index=-1, reduction='none')

def my_model_function():
    return MyModel()

def GetInput():
    # Generate random input matching the shape and dtype from the issue
    return torch.rand(2, 12, 2, dtype=torch.float32)

