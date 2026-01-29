# torch.rand(1, 2, 7, 7, dtype=torch.double)
import torch
import torch.nn as nn
import torch.nn.functional as F

class MyModel(nn.Module):
    def forward(self, x):
        B = x.shape[0]
        samples = x.new(B, 2, 2).uniform_()
        return F.fractional_max_pool2d(
            x, kernel_size=(2, 2), output_size=(3, 3), _random_samples=samples
        )

def my_model_function():
    return MyModel()

def GetInput():
    return torch.randn(1, 2, 7, 7, dtype=torch.double, requires_grad=True)

