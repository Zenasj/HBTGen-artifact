# torch.rand(1, 1, 3, 3, dtype=torch.float32)
import torch
from torch import nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(1, 1, kernel_size=3)
        self.conv.bias.data.fill_(-10.0)  # Ensure negative activations as in the issue example
        self.relu_inplace = nn.ReLU(inplace=True)
        self.relu_noninplace = nn.ReLU(inplace=False)
    
    def forward(self, x):
        conv_out = self.conv(x)
        out1 = self.relu_inplace(conv_out)  # Problematic path (in-place ReLU)
        out2 = self.relu_noninplace(conv_out)  # Fixed path (non-inplace ReLU)
        # Return 1.0 if outputs differ, 0.0 otherwise (tensor for compatibility with torch.compile)
        return torch.tensor(not torch.allclose(out1, out2), dtype=torch.float32)

def my_model_function():
    return MyModel()

def GetInput():
    return torch.rand(1, 1, 3, 3, dtype=torch.float32)

