# torch.rand(B, 3, 28, 28, dtype=torch.float32)  # Inferred input shape for a simple CNN
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.fc = nn.Linear(16 * 28 * 28, 10)  # Example FC layer for classification
        # Atomic operations (like atomic_add) are assumed to be handled via backend-specific implementations
        # (e.g., inductor's C++ codegen), which is the core issue discussed in the GitHub thread

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = x.view(x.size(0), -1)  # Flatten for FC layer
        return self.fc(x)

def my_model_function():
    # Returns a simple CNN model with a structure that may trigger C++ backend compilation
    return MyModel()

def GetInput():
    # Generates a random input tensor matching the expected shape (B=1 for simplicity)
    batch_size = 1
    return torch.rand(batch_size, 3, 28, 28, dtype=torch.float32)

