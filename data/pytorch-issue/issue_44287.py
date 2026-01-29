# torch.rand(B, C, H, W, dtype=torch.float32)  # Add a comment line at the top with the inferred input shape (B, 3, 224, 224)

import torch
import torch.nn as nn
import torchvision.transforms as transforms

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.model = torch.hub.load('pytorch/vision:v0.6.0', 'alexnet', pretrained=True)
        self.model.eval()

    def forward(self, x):
        return self.model(x)

def my_model_function():
    # Return an instance of MyModel, include any required initialization or weights
    return MyModel()

def GetInput():
    # Return a random tensor input that matches the input expected by MyModel
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    transform = transforms.Compose([
        transforms.ToTensor(),
        normalize
    ])
    # Generate a random image tensor with shape (1, 3, 224, 224)
    image = torch.rand(1, 3, 224, 224, dtype=torch.float32)
    return image

