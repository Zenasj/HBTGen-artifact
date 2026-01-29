# torch.rand(B, 3, 224, 224, dtype=torch.float16)  # Assuming input is a 3-channel image tensor in FP16
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        # Define a simple CNN architecture with layers that may trigger fuser issues in FP16
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool(x)
        x = self.conv2(x)
        return x

def my_model_function():
    # Return model in FP16 to simulate problematic scenario
    model = MyModel()
    model = model.half()  # Convert to half-precision
    return model

def GetInput():
    # Generate FP16 input tensor matching expected shape (B=1, C=3, H=224, W=224)
    return torch.rand(1, 3, 224, 224, dtype=torch.float16)

