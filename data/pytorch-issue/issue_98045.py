import torch
import torch.nn as nn
import torch.nn.functional as F

# torch.rand(B, C, H, W, dtype=torch.float32)  # Input shape: (batch, channels, height, width)
class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 768, kernel_size=3, stride=2)  # Example layer to generate intermediate features
        self.size_factor = nn.Parameter(torch.tensor(0.5))  # Parameter causing dynamic size computation

    def forward(self, x):
        x = self.conv(x)
        h, w = x.shape[2], x.shape[3]
        # Compute new size using parameter (triggers Dynamo issue due to FakeTensor in computation)
        new_h = int(h * self.size_factor.item())  # .item() creates data dependency
        new_w = int(w * self.size_factor.item())
        new_size = (new_h, new_w)
        x = F.interpolate(x, size=new_size, mode='bilinear', align_corners=True)
        return x.mean(dim=(2, 3))  # Dummy output to complete forward pass

def my_model_function():
    # Returns an instance with initialized parameters
    return MyModel()

def GetInput():
    # Returns a 4D tensor matching the expected input shape (B, C, H, W)
    return torch.rand(1, 3, 224, 224, dtype=torch.float32)  # Example input dimensions

