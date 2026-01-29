# torch.rand(1, 3, 224, 224, dtype=torch.float32)
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.pad = nn.ReflectionPad2d(1)
        
    def forward(self, input):
        x_in = input.sub(127.)  # Scalar subtraction causing ONNX export issue in older PyTorch versions
        return self.pad(x_in)

def my_model_function():
    return MyModel()

def GetInput():
    return torch.randn(1, 3, 224, 224, dtype=torch.float32)

