# torch.rand(B, C, H, W, dtype=torch.float16)
import torch
import torch.nn as nn
from torchvision.models import resnet18

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = resnet18(pretrained=False)

    def forward(self, x):
        return self.model(x)

def my_model_function():
    model = MyModel()
    model = model.cuda().half()
    model.eval()
    return model

def GetInput():
    return torch.rand(8, 3, 224, 224, dtype=torch.float16, device="cuda")

