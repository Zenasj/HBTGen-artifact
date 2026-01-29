# torch.rand(B, C, H, W, dtype=torch.float32)
import torch
import torchvision.models as models
from torch import nn

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.model = models.alexnet()  # Use AlexNet as described in the issue

    def forward(self, x):
        return self.model(x)

def my_model_function():
    model = MyModel()
    model.to('cuda')  # Replicate error scenario with CUDA
    return model

def GetInput():
    return torch.rand(16, 3, 224, 224, device='cuda')  # Match input shape and device

