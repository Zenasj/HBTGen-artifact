# torch.rand(1, 3, 224, 224, dtype=torch.float32)
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu6 = nn.ReLU6()
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1, groups=32)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(64, 1000)  # Simplified classifier

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu6(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu6(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu6(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        return self.fc(x)

def my_model_function():
    # Returns a traced version of MyModel to simulate TorchScript behavior
    model = MyModel()
    example_input = torch.rand(1, 3, 224, 224)
    traced_model = torch.jit.trace(model, example_input)
    return traced_model

def GetInput():
    # Returns input matching the traced model's requirements
    return torch.rand(1, 3, 224, 224, dtype=torch.float32).cuda()

