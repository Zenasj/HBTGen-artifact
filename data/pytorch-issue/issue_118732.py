# torch.rand(B, C, H, W, dtype=torch.float32)  # Assumed input shape (B=1, C=3, H=224, W=224)
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        # Simple CNN structure inferred due to lack of explicit model details in the issue
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.fc = nn.Linear(32 * 56 * 56, 10)  # 56x56 from 224/2/2 downscaling

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out

def my_model_function():
    # Returns a simple CNN instance with random weights
    return MyModel()

def GetInput():
    # Returns a random tensor matching the assumed input shape
    return torch.rand(1, 3, 224, 224, dtype=torch.float32)

