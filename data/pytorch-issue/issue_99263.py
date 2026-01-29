# torch.rand(4, 3, 224, 224, dtype=torch.float32)
import torch
import torchvision.models as models
from torch import nn

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.resnet = models.resnet152(pretrained=True)
        # Freeze parameters as per original code
        for param in self.resnet.parameters():
            param.requires_grad = False
    
    def forward(self, x):
        return self.resnet(x)

def my_model_function():
    model = MyModel()
    model.eval()  # Ensure model is in inference mode
    return model

def GetInput():
    return torch.rand(4, 3, 224, 224, dtype=torch.float32)

