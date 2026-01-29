# torch.rand(B, C, H, W, dtype=...)  # B: batch_size, C: channels, H: height, W: width
import torch
import torch.nn as nn
import torchvision.models as models

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.resnet = models.resnet18()
        self.resnet.fc = nn.Linear(512, 100)

    def forward(self, x):
        return self.resnet(x)

def my_model_function():
    # Return an instance of MyModel, include any required initialization or weights
    return MyModel()

def GetInput():
    # CIFAR-100 images are 32x32 with 3 channels
    # Batch size is 64
    return torch.rand(64, 3, 32, 32, dtype=torch.float32)

