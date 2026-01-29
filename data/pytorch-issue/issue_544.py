# torch.rand(1, 256, 120, 192, dtype=torch.float32).cuda()  # Inferred input shape from the issue's bad_sizes[2]

import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.conv = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
    
    def forward(self, x):
        return self.conv(x)

def my_model_function():
    model = MyModel()
    model.cuda()  # Place model on CUDA to trigger the reported error
    return model

def GetInput():
    return torch.randn(1, 256, 120, 192, dtype=torch.float32).cuda()

