# torch.rand(B, C, H, W, dtype=torch.float32)  # Input shape assumed as (1, 3, 256, 256) based on the minimal repro code
import torch
import torch.nn as nn
import torch.nn.functional as F

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
    
    def forward(self, x):
        # Apply nearest interpolation with scale_factor=2
        return F.interpolate(x, scale_factor=2, mode='nearest')

def my_model_function():
    # Returns the model instance with necessary configurations
    model = MyModel()
    return model

def GetInput():
    # Generates a random input tensor matching the expected shape
    return torch.rand(1, 3, 256, 256, dtype=torch.float32)

# Override ONNX symbolic function to fix Resize node creation
# This must be placed before calling torch.onnx.export
import torch.onnx.symbolic_opset11 as onnx_symbolic

def fixed_upsample_nearest2d(g, input, output_size, *args):
    scales = g.op("Constant", value_t=torch.tensor([1.0, 1.0, 2.0, 2.0], dtype=torch.float32))
    empty_tensor = g.op("Constant", value_t=torch.tensor([], dtype=torch.float32))
    return g.op("Resize", input, empty_tensor, scales, mode_s="nearest", nearest_mode_s="floor")

onnx_symbolic.upsample_nearest2d = fixed_upsample_nearest2d

