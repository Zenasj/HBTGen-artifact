# torch.rand(B, 3, 224, 224, dtype=torch.float32)  # Inferred input shape for a typical CNN
import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )

    def forward(self, x):
        # Checkpointing applied to reduce memory usage during training
        x = checkpoint(self.layer1, x)
        x = checkpoint(self.layer2, x)
        return x

def my_model_function():
    # Returns a model instance using checkpointing with proper detach() handling
    return MyModel()

def GetInput():
    # Generate random input tensor matching the model's expected dimensions
    return torch.rand(2, 3, 224, 224, dtype=torch.float32)

