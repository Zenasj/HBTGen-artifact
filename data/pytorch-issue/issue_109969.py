import torch
import torchvision.models

# torch.rand(B, 3, 224, 224, dtype=torch.float32)
class MyModel(torch.nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.model = torchvision.models.resnet18(weights='IMAGENET1K_V1')

    def forward(self, x):
        return self.model(x)

def my_model_function():
    return MyModel()

def GetInput():
    return torch.rand(32, 3, 224, 224, dtype=torch.float32)

