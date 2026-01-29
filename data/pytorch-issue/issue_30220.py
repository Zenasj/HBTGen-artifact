# torch.rand(1, 3, 105, 105, dtype=torch.float)
import torch
import torch.nn as nn
import torchvision.transforms as transforms

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    
    def forward(self, x):
        return self.normalize(x)

def my_model_function():
    return MyModel()

def GetInput():
    return torch.rand(1, 3, 105, 105, dtype=torch.float)

