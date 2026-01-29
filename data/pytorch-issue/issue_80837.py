# torch.rand(1, 1, 321, 201, dtype=torch.float32)
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.conv = nn.Conv2d(
            1,
            128,
            kernel_size=(5, 2),
            stride=(2, 1),
            padding=(0, 1),
            dilation=(1, 1),
            groups=1,
            bias=True,
            padding_mode='zeros'
        )
    
    def forward(self, x):
        return self.conv(x)

def my_model_function():
    return MyModel()

def GetInput():
    t = torch.rand([1, 2, 321, 201, 1])
    t = t.transpose(1, 4)  # Swap dimensions 1 and 4
    t2 = t[..., 0]         # Slice to remove last dimension (size 2)
    return t2

