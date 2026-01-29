# torch.rand(20, dtype=torch.half).cuda()
import torch
from torch import nn

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
    
    def forward(self, x):
        # Convert to FP32 to avoid "multinomial_kernel_cuda" error for Half dtype
        x_float = x.float()
        return torch.distributions.Categorical(logits=x_float).sample()

def my_model_function():
    return MyModel()

def GetInput():
    # Generates a random FP16 tensor matching the input expected by MyModel
    return torch.randn(20, dtype=torch.half).cuda()

