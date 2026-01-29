import torch
import torch.nn as nn
import torch.nn.functional as F

# torch.rand(B, 784, dtype=torch.float32)
class MyModel(nn.Module):
    def __init__(self, input_size=784, num_classes=10):
        super(MyModel, self).__init__()
        self.fc1 = nn.Linear(input_size, 50)
        self.fc2 = nn.Linear(50, num_classes)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

def my_model_function():
    return MyModel()

def GetInput():
    return torch.randn(64, 784, dtype=torch.float32)

