# torch.rand(1, 3, 200, 300, dtype=torch.float32)
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        # Simulate a simplified detection model structure using ModuleDict to trigger GetAttr nodes
        self.layers = nn.ModuleDict({
            'backbone': nn.Sequential(
                nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
                nn.Conv2d(64, 128, kernel_size=3, padding=1),
                nn.ReLU()
            ),
            'head': nn.Sequential(
                nn.AdaptiveAvgPool2d((7, 7)),
                nn.Flatten(),
                nn.Linear(128*7*7, 10)  # Mock output layer
            )
        })

    def forward(self, x):
        x = self.layers['backbone'](x)  # Explicit GetAttr access pattern
        return self.layers['head'](x)

def my_model_function():
    # Create model instance matching torchvision detection model pattern
    model = MyModel()
    model.eval()  # Ensure evaluation mode as in original reproduction
    return model

def GetInput():
    # Generate input matching (N, C, H, W) with min_size=200 and max_size=300
    return torch.rand(1, 3, 200, 300, dtype=torch.float32)

