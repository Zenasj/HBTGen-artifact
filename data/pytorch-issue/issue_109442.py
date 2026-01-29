# torch.rand(B, 1, 3, 224, 224, dtype=torch.float32)  # Assuming a batch size of 1 and 3-channel input images of 224x224
import torch
import torch.nn as nn
import torch.nn.functional as F

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5, padding=2)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5, padding=2)
        self.fc1 = nn.Linear(16 * 56 * 56, 10)  # 56x56 from pooling steps

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)  # Flatten all dimensions except batch
        x = self.fc1(x)
        return x

def my_model_function():
    # Returns a model instance with default initialization
    return MyModel()

def GetInput():
    # Generate a random input tensor matching the model's expected input shape
    return torch.rand(1, 3, 224, 224, dtype=torch.float32)

