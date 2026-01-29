# torch.rand(B, 3, 224, 224, dtype=torch.float32)
import torch
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
        
    def forward(self, x):
        return self.model(x)

def my_model_function():
    return MyModel()

def GetInput():
    B = 2  # Matches batch_size from issue's example
    input_tensor = torch.rand(B, 3, 224, 224, dtype=torch.float32)
    # Apply normalization from user's transforms
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    normalized = (input_tensor - mean) / std
    return normalized

