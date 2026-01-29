# torch.rand(B, 3, 224, 224, dtype=torch.float32)  # Inferred input shape based on common CNN usage
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        # Basic CNN structure to match FSDP usage context
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(32 * 56 * 56, 128)  # 56x56 from 224/2²
        self.fc2 = nn.Linear(128, 10)
        
    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = torch.flatten(x, 1)
        x = self.relu(self.fc1(x))
        return self.fc2(x)

def my_model_function():
    # Returns a basic CNN instance suitable for FSDP wrapping
    return MyModel()

def GetInput():
    # Generates a batch of 3-channel 224x224 images
    B = 4  # Batch size
    return torch.rand(B, 3, 224, 224, dtype=torch.float32)

