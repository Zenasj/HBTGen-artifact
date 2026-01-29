# torch.rand(10, 10, 10, requires_grad=True) ← Add a comment line at the top with the inferred input shape
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
    
    def forward(self, x):
        # Apply FFT and take the real part
        x = torch.fft.fftn(x)
        x = x.real
        return x.mean()

def my_model_function():
    # Return an instance of MyModel, include any required initialization or weights
    return MyModel()

def GetInput():
    # Return a random tensor input that matches the input expected by MyModel
    return torch.rand(10, 10, 10, requires_grad=True)

