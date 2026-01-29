# torch.rand(B, C, H, W, dtype=torch.int32)  # Inferred input shape for the model
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()

    def forward(self, x: torch.Tensor):
        # Cast to float for compatibility with older ONNX opsets
        x = x.float()
        # Perform the max operation
        output = torch.max(x, x + 1)
        # Cast back to the original type
        return output.to(x.dtype)

def my_model_function():
    # Return an instance of MyModel, include any required initialization or weights
    return MyModel()

def GetInput():
    # Return a random tensor input that matches the input expected by MyModel
    return torch.randint(0, 10, (3, 3), dtype=torch.int32)

