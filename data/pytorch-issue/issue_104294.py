# torch.rand(B, 3, 224, 224, dtype=torch.float32)  # Assumed input shape for image classification benchmarks
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        # Basic CNN structure inferred for benchmark compatibility
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.fc1 = nn.Linear(32 * 56 * 56, 100)  # 224/2 = 112; 112/2=56 (after two pools)
        self.fc2 = nn.Linear(100, 10)  # Assuming 10-class classification

    def forward(self, x):
        x = self.pool(self.relu(self.bn1(self.conv1(x))))
        x = self.pool(self.relu(self.bn2(self.conv2(x))))
        x = torch.flatten(x, 1)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

def my_model_function():
    # Returns a basic CNN model compatible with benchmark requirements
    return MyModel()

def GetInput():
    # Generates random input tensor matching expected shape
    B = 4  # Batch size placeholder
    return torch.rand(B, 3, 224, 224, dtype=torch.float32)

