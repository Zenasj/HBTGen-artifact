# torch.rand(B, 3, 299, 299, dtype=torch.float32)
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        # Simplified InceptionV3 layers structure to replicate the issue scenario
        self.layers = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, stride=2, ceil_mode=True),
            nn.Conv2d(64, 80, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(80, 192, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, stride=2, ceil_mode=True),
            # ... (additional layers omitted for brevity)
        )
        # Placeholder for full model comparison (required by Special Requirement 2)
        self.full_model = nn.Sequential(
            self.layers,
            nn.AdaptiveAvgPool2d((8,8)),
            nn.Flatten(),
            nn.Linear(192*8*8, 1000)  # Matches InceptionV3 output size
        )

    def forward(self, x):
        # Simulate the issue scenario where layers submodule is called independently
        # (while also allowing full model comparison as per requirement)
        return self.layers(x)  # Fails with CUDA fuser enabled

def my_model_function():
    return MyModel()

def GetInput():
    return torch.randn(1, 3, 299, 299, dtype=torch.float32, device='cuda')

