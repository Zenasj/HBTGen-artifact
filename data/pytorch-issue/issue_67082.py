import torch
import torch.nn as nn
import torch.nn.functional as F

# torch.rand(B=1, C=3, H=224, W=224, dtype=torch.float32)  # Inferred input shape for a typical image model
class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)  # Example layer compatible with ONNX export
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 53 * 53, 120)  # Calculated based on input size 224x224
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)  # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

def my_model_function():
    return MyModel()  # Returns a basic model instance for ONNX export testing

def GetInput():
    return torch.rand(1, 3, 224, 224, dtype=torch.float32)  # Matches the model's expected input dimensions

