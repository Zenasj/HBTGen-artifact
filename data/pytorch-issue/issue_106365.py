import torch
import torch.nn as nn

# torch.rand(B, 3, 32, 32, dtype=torch.float32)  # Assumed input shape based on common CNN use cases
class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.fc = nn.Linear(16 * 32 * 32, 10)  # Matches input shape (32x32) after convolution
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

def my_model_function():
    # Returns a simple CNN model instance
    return MyModel()

def GetInput():
    # Returns a random input tensor matching the model's expected input shape
    return torch.rand(2, 3, 32, 32, dtype=torch.float32)

