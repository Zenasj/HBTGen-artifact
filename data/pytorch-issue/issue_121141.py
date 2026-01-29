# torch.rand(B, 3, 224, 224, dtype=torch.float32)
import torch
import torchvision.models as models
from torch import nn

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.model = models.resnet18()
    
    def forward(self, x):
        return self.model(x)

def my_model_function():
    model = MyModel()
    model = model.cuda()
    return model

def GetInput():
    return torch.rand(1, 3, 224, 224, dtype=torch.float32).cuda()

