# torch.rand(B, C, H, W, dtype=torch.float32)
import torch
from torch import nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        # Create CUDA event with interprocess=True to match the issue's context
        self.event = torch.cuda.Event(enable_timing=False, interprocess=True)
    
    def forward(self, x):
        # Record event on current stream during forward pass
        stream = torch.cuda.current_stream()
        stream.record_event(self.event)
        return x  # Pass-through for compilation compatibility

def my_model_function():
    return MyModel()

def GetInput():
    # Generate random input tensor with common shape
    return torch.rand(1, 3, 224, 224, dtype=torch.float32)

