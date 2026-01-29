import torch
from torch import nn

# torch.rand(1, 3, 32, 32, dtype=torch.float32)
class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.layer1 = nn.Conv2d(3, 32, 1, 1)
    
    def forward(self, x):
        return self.layer1(x)

def my_model_function():
    model = MyModel()
    model.eval()
    model.cuda()
    return model

def GetInput():
    shape = [1, 3, 32, 32]
    strides = (3072, 1, 96, 3)  # Non-contiguous strides to reproduce the issue
    x = torch.randn(shape).as_strided(shape, strides).cuda()
    return x

