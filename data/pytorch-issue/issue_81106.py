# torch.rand(B, C, H, W, dtype=torch.float32)  # Input shape: 8x3x256x256, float32
import torch
import torchvision.transforms.functional as F
from torch import nn

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.conv1 = nn.Conv2d(1, 8, kernel_size=5, stride=1, bias=False)

    def forward(self, x):
        print('before', x.device, x.dtype, x.layout, x.is_contiguous())
        x = F.rgb_to_grayscale(x, num_output_channels=1)
        print('after', x.device, x.dtype, x.layout, x.is_contiguous())
        x = self.conv1(x)
        return x

def my_model_function():
    return MyModel()

def GetInput():
    return torch.rand(8, 3, 256, 256, dtype=torch.float32)

