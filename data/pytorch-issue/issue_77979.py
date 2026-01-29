# torch.rand(B, C, L, dtype=torch.float32)  # Input shape (Batch, Channels, Length)
import torch
import torch.nn as nn
import torch.nn.functional as F

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        # Define running statistics buffers explicitly to ensure ONNX export compatibility
        self.register_buffer('running_mean', torch.zeros(2))  # 2 channels
        self.register_buffer('running_var', torch.ones(2))
        
    def forward(self, x):
        # Use explicit parameters to avoid ONNX runtime errors with DML provider
        return F.instance_norm(
            x,
            use_input_stats=False,  # Force use of precomputed running stats
            running_mean=self.running_mean,
            running_var=self.running_var
        )

def my_model_function():
    # Returns an instance with buffers initialized as in the working example
    return MyModel()

def GetInput():
    # Matches the input shape expected by MyModel (B=1, C=2, L=2)
    return torch.randn(1, 2, 2)

