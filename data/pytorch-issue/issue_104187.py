# torch.rand(B, 10, dtype=torch.float32)
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(10, 20)
        self.bn = nn.BatchNorm1d(20)
        self.fc2 = nn.Linear(20, 10)
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.bn(x)
        x = torch.relu(x)
        x = self.fc2(x)
        return x
    
    def reset_parameters(self):
        # No-op here as submodules handle their own parameters/buffers
        # Linear and BatchNorm layers have their own reset_parameters()
        pass

def my_model_function():
    # Returns a model instance with standard initialization
    model = MyModel()
    return model

def GetInput():
    # Generates a random input tensor matching the model's expected input shape
    return torch.rand(2, 10, dtype=torch.float32)

