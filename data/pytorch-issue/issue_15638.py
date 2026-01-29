# torch.rand(B, C, H, W, dtype=torch.float32)  # Assumed input shape (e.g., 4x3x224x224 image batch)
import torch
import torch.nn as nn
import torch.nn.functional as F

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        # Minimal CNN structure for demonstration (shape assumptions: input 3x224x224)
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 53 * 53, 120)  # 53 = (224-4)/2/2 ≈ 53 (approximate for simplicity)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 53 * 53)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

def my_model_function():
    # Returns a simple CNN model instance
    return MyModel()

def GetInput():
    # Returns a random tensor matching the assumed input shape (B=4, C=3, H=224, W=224)
    return torch.rand(4, 3, 224, 224, dtype=torch.float32)

