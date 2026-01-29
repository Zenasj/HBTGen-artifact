# torch.rand(B, 3, 224, 224, dtype=torch.float32)
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        # Fused model structure mimicking cascaded topologies (Model1, Model2, Model3)
        self.model1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        self.model2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        self.model3 = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 56 * 56, 1000)
        )

    def forward(self, x):
        x = self.model1(x)
        x = self.model2(x)
        x = self.model3(x)
        return x

def my_model_function():
    # Returns fused model instance with default initialization
    return MyModel()

def GetInput():
    # Generates input matching the cascaded topology's requirements
    B = 1  # Batch size
    return torch.rand(B, 3, 224, 224, dtype=torch.float32)

