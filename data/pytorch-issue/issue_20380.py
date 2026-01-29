# torch.rand(B, 3, 32, 32, dtype=torch.float32)
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.fc = nn.Linear(16*32*32, 10)  # Output 10-class logits
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = x.view(x.size(0), -1)  # Flatten for linear layer
        x = self.fc(x)
        return x

def my_model_function():
    # Returns a simple CNN model compatible with distributed training
    return MyModel()

def GetInput():
    # Generate a random input tensor matching the model's expected dimensions
    B = 4  # Example batch size
    return torch.rand(B, 3, 32, 32, dtype=torch.float32)

