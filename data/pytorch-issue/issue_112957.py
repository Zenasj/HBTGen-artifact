# torch.rand(B, C, H, W, dtype=torch.float32)  # Inferred input shape: (B, 3, 200, 200)
import torch
import torch.nn as nn
import torchvision.models as models

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        # Load the pre-trained MobileNetV2 and use only the features part
        mobilenet = models.mobilenet_v2(weights="DEFAULT")
        self.features = mobilenet.features[:-1].eval()
        self.adaptive_pool = nn.AdaptiveAvgPool2d(1)  # Corrected to AdaptiveAvgPool2d

    def forward(self, x):
        x = self.features(x)
        x = self.adaptive_pool(x)
        return x

def my_model_function():
    # Return an instance of MyModel
    return MyModel()

def GetInput():
    # Return a random tensor input that matches the input expected by MyModel
    B, C, H, W = 64, 3, 200, 200
    return torch.rand(B, C, H, W, dtype=torch.float32, device='cuda' if torch.cuda.is_available() else 'cpu')

