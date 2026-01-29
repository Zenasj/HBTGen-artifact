# torch.rand(1, 3, 224, 224, dtype=torch.float32)
import torch
import torchvision

class MyModel(torch.nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.model = torchvision.models.resnet50()

    def forward(self, x):
        return self.model(x)

def my_model_function():
    return MyModel()

def GetInput():
    return torch.rand(1, 3, 224, 224, dtype=torch.float32)

