# torch.rand(1, 3, 224, 224, dtype=torch.float32, device='cuda')
import torch
import torchvision.models as models
from torch import nn

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.model = models.resnet18(pretrained=True)
        self.model.to('cuda')
        self.model.eval()

    def forward(self, x):
        return self.model(x)

def my_model_function():
    return MyModel()

def GetInput():
    return torch.rand(1, 3, 224, 224, dtype=torch.float32, device='cuda')

