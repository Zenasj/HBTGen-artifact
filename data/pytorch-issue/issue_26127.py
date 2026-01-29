# torch.rand(B, C, H, W, dtype=torch.float32)  # Add a comment line at the top with the inferred input shape

import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()

    @torch.jit.script
    def yolo_xy_dynsize_helper(height, width):
        """The input shape could be dynamic"""
        i = torch.arange(width, dtype=torch.float32)
        j = torch.arange(height, dtype=torch.float32)
        return i, j

    def forward(self, xy):
        height, width = xy.shape[2], xy.shape[3]
        # When tracing (for onnx export this is Tensor)
        if not isinstance(height, torch.Tensor):
            height, width = torch.tensor(height), torch.tensor(width)
        i, j = self.yolo_xy_dynsize_helper(height, width)
        return i, j

def my_model_function():
    # Return an instance of MyModel, include any required initialization or weights
    return MyModel()

def GetInput():
    # Return a random tensor input that matches the input expected by MyModel
    B, C, H, W = 1, 3, 4, 5  # Example batch size, channels, height, and width
    return torch.rand(B, C, H, W, dtype=torch.float32)

