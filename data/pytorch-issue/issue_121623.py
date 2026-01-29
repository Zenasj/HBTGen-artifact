# torch.rand(1, 6, 640, 640, dtype=torch.float32)  # Add a comment line at the top with the inferred input shape

import torch
import torch.nn as nn
from torchvision import transforms
from torch.hub import load

class SepratorLayers(nn.Module):
    def __init__(self):
        super(SepratorLayers, self).__init__()

    def forward(self, x):
        img1, img2 = x[:, :3, :, :], x[:, 3:, :, :]
        return img1, img2

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.seprate = SepratorLayers()
        self.vgg11 = load('pytorch/vision:v0.10.0', 'vgg11', pretrained=True)
        self.vgg19 = load('pytorch/vision:v0.10.0', 'vgg19', pretrained=True)

        self.transform = transforms.Compose([
            transforms.Resize((224, 224), antialias=False),  # Set antialias to False to avoid the error
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])

    def forward(self, x):
        img1, img2 = self.seprate(x)
        img1 = self.transform(img1)
        img2 = self.transform(img2)
        x1 = self.vgg11(img1)
        x2 = self.vgg19(img2)
        return x1, x2

def my_model_function():
    # Return an instance of MyModel, include any required initialization or weights
    return MyModel()

def GetInput():
    # Return a random tensor input that matches the input expected by MyModel
    return torch.rand(1, 6, 640, 640, dtype=torch.float32)

