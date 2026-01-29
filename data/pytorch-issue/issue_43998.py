# torch.rand(B, 3, 224, 224, dtype=torch.float)
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.fc = nn.Linear(3 * 224 * 224, 1024)  # Simplified ResNet-like output layer
    
    def forward(self, x):
        # Flattens input tensor for linear layer
        x = x.view(x.size(0), -1)
        return self.fc(x)

def my_model_function():
    return MyModel()

def GetInput():
    # Batch size 128 as in the test case
    return torch.randn(128, 3, 224, 224, dtype=torch.float)

