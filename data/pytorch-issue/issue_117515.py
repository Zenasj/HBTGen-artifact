# torch.randn(5, 3, 224, 224, dtype=torch.float32)
import torch
import torchvision.models as models

class MyModel(torch.nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.resnet = models.resnet18()  # Use torchvision's ResNet18 as base model

    def forward(self, x):
        return self.resnet(x)

def my_model_function():
    # Returns the pre-defined ResNet18 model
    return MyModel()

def GetInput():
    # Generate random input matching ResNet18's expected dimensions
    return torch.randn(5, 3, 224, 224, dtype=torch.float32)

