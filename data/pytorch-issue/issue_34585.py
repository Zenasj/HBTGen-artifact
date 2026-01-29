# torch.rand(5, 5, dtype=torch.float32)
import torch
import torch.nn as nn
import torch.nn.functional as F

class MyModel(nn.Module):
    class OriginalTraceCheck(nn.Module):
        def forward(self, x):
            return F.softmax(-100 * x, dim=-2)  # Original problematic axis
    
    class FixedTraceCheck(nn.Module):
        def forward(self, x):
            return F.softmax(-100 * x, dim=1)  # Fixed axis for ONNX compatibility
    
    def __init__(self):
        super().__init__()
        self.original = self.OriginalTraceCheck()  # Original model with problematic axis
        self.fixed = self.FixedTraceCheck()        # Fixed model for ONNX
    
    def forward(self, x):
        # Return outputs from both models to compare their behavior
        return self.original(x), self.fixed(x)

def my_model_function():
    return MyModel()

def GetInput():
    return torch.randn(5, 5, dtype=torch.float32)

