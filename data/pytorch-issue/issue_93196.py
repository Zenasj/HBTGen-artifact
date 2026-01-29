# torch.rand(16, 3, 224, 224, dtype=torch.float32)
import torch
import torchvision.models as models

class MyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model = models.resnet18()

    def forward(self, x):
        return self.model(x)

def my_model_function():
    return MyModel()

def GetInput():
    return torch.randn(16, 3, 224, 224, dtype=torch.float32)

