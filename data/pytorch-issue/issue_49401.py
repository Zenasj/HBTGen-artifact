# torch.rand(B, C, H, W, dtype=torch.float32)  # B=1568, C=3, H=64, W=64

import torch
import torch.nn as nn
import torchvision.models as models

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.resnet = models.resnet18(pretrained=True, progress=True)
    
    def forward(self, x):
        return self.resnet(x)

def my_model_function():
    # Return an instance of MyModel, include any required initialization or weights
    return MyModel()

def GetInput():
    # Return a random tensor input that matches the input expected by MyModel
    batch_size = 1568
    channels = 3
    height = 64
    width = 64
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    return torch.rand(batch_size, channels, height, width, dtype=torch.float32).to(device)

