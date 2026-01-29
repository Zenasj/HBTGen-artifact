# torch.rand(B, 3, H, W, dtype=torch.float32)
import torch
from torch import nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.pool = nn.AvgPool2d(2)
        
    def forward(self, x):
        x = self.conv1(x)
        h, w = x.shape[2], x.shape[3]
        # Critical FloorDiv operation causing dynamic shape path issue
        new_h = (h + 1) // 2  # Emulates division pattern seen in the PR's loop index
        new_w = (w + 1) // 2
        x = x[:, :, :new_h, :new_w]
        x = self.pool(x)
        return x

def my_model_function():
    return MyModel()

def GetInput():
    # Dynamic input shape (B=1, C=3, H=224, W=224) matching common image inputs
    return torch.rand(1, 3, 224, 224, dtype=torch.float32)

