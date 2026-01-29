# torch.rand(6, 192, 30, 96, dtype=torch.float32, device="cuda")
import torch
from torch import nn

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.conv_transpose = nn.ConvTranspose2d(
            in_channels=192,
            out_channels=384,
            kernel_size=(2, 2),
            stride=2
        )
    
    def forward(self, x):
        return self.conv_transpose(x)

def my_model_function():
    model = MyModel()
    model.to("cuda")  # Matches device used in original issue's example
    return model

def GetInput():
    return torch.rand(6, 192, 30, 96, dtype=torch.float32, device="cuda")

