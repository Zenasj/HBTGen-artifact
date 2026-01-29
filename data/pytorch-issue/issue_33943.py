# torch.rand(B, 100, dtype=torch.float32)  # Inferred input shape for a simple linear model
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.fc = nn.Linear(100, 10)  # Matches input shape's 100 features
    
    def forward(self, x):
        return self.fc(x)

def my_model_function():
    # Returns a simple linear model with random initialization
    return MyModel()

def GetInput():
    B = 4  # Arbitrary batch size, can be adjusted based on MPI world size
    return torch.rand(B, 100, dtype=torch.float32)  # Matches the input shape comment

