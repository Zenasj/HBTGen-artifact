# torch.rand(B, 3, 224, 224, dtype=torch.float32)
import torch
import torch.nn as nn

class Features(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv0 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.relu0 = nn.ReLU(inplace=True)
    
    def forward(self, x):
        x = self.conv0(x)
        x = self.relu0(x)
        return x

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = Features()
        # Example structure; actual layers may vary based on original model
        self.classifier = nn.Linear(64 * 224 * 224, 10)  # Placeholder
    
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

def my_model_function():
    return MyModel()

def GetInput():
    B, C, H, W = 1, 3, 224, 224  # Batch size, channels, height, width
    return torch.rand(B, C, H, W, dtype=torch.float32)

