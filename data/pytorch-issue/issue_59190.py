# torch.rand(B, 3, 513, 513, dtype=torch.float32)  # DeeplabV3 expects 3-channel images
import torch
import torch.nn as nn
import torchvision.models as models

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.model = models.segmentation.deeplabv3_resnet50(pretrained=True)
        self.model.eval()  # Ensure eval mode as in the issue example

    def forward(self, x):
        return self.model(x)["out"]  # Output the main prediction from DeeplabV3

def my_model_function():
    # Returns the DeeplabV3 model instance in eval mode
    return MyModel()

def GetInput():
    # Generate a random input tensor matching DeeplabV3's expected input shape
    return torch.rand(1, 3, 513, 513, dtype=torch.float32)

