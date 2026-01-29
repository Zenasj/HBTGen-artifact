# torch.rand(B, C, H, W, dtype=...)  # Inferred input shape: (1, 1, 4, 4) for each tensor in the nested tensor

import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        # Define the model structure
        self.adaptive_avg_pool2d = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        # Apply adaptive average pooling to the nested tensor
        return self.adaptive_avg_pool2d(x)

def my_model_function():
    # Return an instance of MyModel
    return MyModel()

def GetInput():
    # Generate a nested tensor with jagged layout
    tensor1 = torch.rand([1, 1, 4, 4], dtype=torch.float32)
    tensor2 = torch.rand([1, 1, 4, 4], dtype=torch.float32)
    nested_tensor = torch.nested.nested_tensor([tensor1, tensor2], layout=torch.jagged)
    return nested_tensor

