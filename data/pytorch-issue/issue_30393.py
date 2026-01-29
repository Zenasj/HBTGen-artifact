# torch.rand(1, 3, 544, 1920, dtype=torch.float32)  # Add a comment line at the top with the inferred input shape
import torch
import torch.nn as nn
import torch.nn.functional as F

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.expander = nn.Conv2d(3, 192, 1, 1)
        self.P4_upsampled = nn.Upsample(scale_factor=2, mode='nearest')

    def forward(self, x):
        x = self.expander(x)
        sh = torch.tensor(x.shape[-2:], dtype=torch.int32)  # Use int32 to avoid Cast node in ONNX
        print(sh)
        a = F.interpolate(x, (sh[0] * 2, sh[1] * 2))
        return a

def my_model_function():
    # Return an instance of MyModel, include any required initialization or weights
    return MyModel()

def GetInput():
    # Return a random tensor input that matches the input expected by MyModel
    return torch.rand(1, 3, 544, 1920, dtype=torch.float32)

# The model should be ready to use with `torch.compile(MyModel())(GetInput())`

